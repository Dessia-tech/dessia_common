#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import sys
import warnings
import math
import random
import copy
from functools import reduce
import collections
from copy import deepcopy
import inspect
import json
from typing import List, Union
try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7
import traceback as tb
import dessia_common.typings as dt

from importlib import import_module


JSONSCHEMA_HEADER = {"definitions": {},
                     "$schema": "http://json-schema.org/draft-07/schema#",
                     "type": "object",
                     "required": [],
                     "properties": {}}

TYPING_EQUIVALENCES = {int: 'number', float: 'number',
                       bool: 'boolean', str: 'string'}

TYPES_STRINGS = {int: 'int', float: 'float', bool: 'boolean', str: 'str',
                 list: 'list', tuple: 'tuple', dict: 'dict'}

SEQUENCE_TYPINGS = ['List', 'Sequence', 'Iterable']

_FORBIDDEN_ARGNAMES = ['self', 'cls', 'progress_callback', 'return']

TYPES_FROM_STRING = {'unicode': str, 'str': str, 'float': float,
                     'int': int, 'bool': bool}


class ExceptionWithTraceback(Exception):
    def __init__(self, message, traceback_=''):
        self.message = message
        self.traceback = traceback_

    def __str__(self):
        return '{}\nTraceback:\n{}'.format(self.message, self.traceback)


class DeepAttributeError(ExceptionWithTraceback, AttributeError):
    pass


class ModelError(Exception):
    pass


class ConsistencyError(Exception):
    pass


class SerializationError(Exception):
    pass


class DeserializationError(Exception):
    pass


# DEPRECATED_ATTRIBUTES = {'_editable_variss' : '_allowed_methods'}
def deprecated(use_instead=None):
    def decorated(function):
        def wrapper(*args, **kwargs):
            deprecation_warning(function.__name__, 'Function', use_instead)
            print('Traceback : ')
            tb.print_stack(limit=2)
            return function(*args, **kwargs)

        return wrapper

    return decorated


def deprecation_warning(name, object_type, use_instead=None):
    warnings.simplefilter('once', DeprecationWarning)
    msg = "\n\n{} {} is deprecated.\n".format(object_type, name)
    msg += "It will be removed in a future version.\n"
    if use_instead is not None:
        msg += "Use {} instead.\n".format(use_instead)
    warnings.warn(msg, DeprecationWarning)
    return msg


class DessiaObject:
    """
    Base class for Dessia's platform compatible objects.
    Gathers generic methods and attributes

    :cvar bool _standalone_in_db:
        Indicates wether class objects should be independant
        in database or not.
        If False, object will only exist inside its parent.
    :cvar bool _eq_is_data_eq:
        Indicates which type of equality check is used:
        strict equality or equality based on data. If False, Python's
        object __eq__ method is used (ie. strict),
        else, user custom data_eq is used (ie. data)
    :cvar List[str] _non_serializable_attributes:
        [Advanced] List of instance attributes that should not
        be part of serialization with to_dict method. These will not
        be displayed in platform object tree, for instance.
    :cvar List[str] _non_data_eq_attributes:
        [Advanced] List of instance attributes that should not
        be part of equality check with data__eq__ method
        (if _eq_is_data_eq is True).
    :cvar List[str] _non_data_hash_attributes:
        [Advanced] List of instance attributes that should not
        be part of hash computation with data__hash__ method
        (if _eq_is_data_eq is True).
    :cvar List[str] _ordered_attributes:
        Documentation not available yet.
    :cvar List[str] _titled_attributes:
        Documentation not available yet.
    :cvar List[str] _init_variables:
        Documentation not available yet.
    :cvar List[str] _export_formats:
        List of all available export formats. Class must define a
        export_[format] for each format in _export_formats
    :cvar List[str] _allowed_methods:
        List of all methods that are runnable from platform.
    :cvar List[str] _whitelist_attributes:
        Documentation not available yet.
    :cvar List[str] _whitelist_attributes: List[str]


    :ivar str name: Name of object.
    :ivar Any **kwargs: Additionnal user metadata
    """
    _standalone_in_db = False
    _non_serializable_attributes = []
    _non_editable_attributes = []
    _non_data_eq_attributes = ['name']
    _non_data_hash_attributes = ['name']
    _ordered_attributes = []
    _titled_attributes = []
    _eq_is_data_eq = True
    
    _init_variables = None
    _export_formats = None
    _allowed_methods = []
    _whitelist_attributes = []

    def __init__(self, name: str = '', **kwargs):
        self.name = name
        for property_name, property_value in kwargs.items():
            setattr(self, property_name, property_value)

    def __hash__(self):
        if self._eq_is_data_eq:
            return self._data_hash()
        else:
            return object.__hash__(self)

    def __eq__(self, other_object):
        if self._eq_is_data_eq:
            if self.__class__.__name__ != other_object.__class__.__name__:
                return False
            if self._data_hash() != other_object._data_hash():
                return False
            return self._data_eq(other_object)
        return object.__eq__(self, other_object)

    def _data_eq(self, other_object):
        if full_classname(self) != full_classname(other_object):
            return False

        eq_dict = {k: v for k, v in self.to_dict().items()
                   if (k not in ['package_version', 'name'])\
                       and (k not in self._non_data_eq_attributes)}
        other_eq_dict = other_object.to_dict()
        

        for key, value in eq_dict.items():
            other_value = other_eq_dict[key]
            if value != other_value:
                return False
        return True

    def _data_hash(self):
        hash_ = 0
        forbidden_keys = (self._non_data_eq_attributes
                          + self._non_data_hash_attributes
                          + ['package_version', 'name'])
        for key, value in self.to_dict().items():
            if key not in forbidden_keys:
                if isinstance(value, list):
                    hash_ += list_hash(value)
                elif isinstance(value, dict):
                    hash_ += dict_hash(value)
                elif isinstance(value, str):
                    hash_ += sum([ord(v) for v in value])
                else:
                    hash_ += hash(value)
        return int(hash_ % 1e5)

    def _data_diff(self, other_object):
        """
        Make a diff between two objects
        returns: different values, missing keys in other object
        """
        missing_keys_in_other_object = []
        diff_values = {}
        
        eq_dict = {k: v for k, v in self.to_dict().items()
                   if (k not in ['package_version', 'name'])\
                       and (k not in self._non_data_eq_attributes)}
        other_eq_dict = other_object.to_dict()

        for key, value in eq_dict.items():
            if not key in other_eq_dict:
                missing_keys_in_other_object.append(key)
            else:                
                other_value = other_eq_dict[key]
                if value != other_value:
                    diff_values[key] = (value, other_value)
                
        return diff_values, missing_keys_in_other_object


    @property
    def full_classname(self):
        return full_classname(self)

    def base_dict(self):
        package_name = self.__module__.split('.')[0]
        if package_name in sys.modules:
            package = sys.modules[package_name]
            if hasattr(package, '__version__'):
                package_version = package.__version__
            else:
                package_version = None
        else:
            package_version = None

        dict_ = {'name': self.name,
                 'object_class': self.__module__ + '.' + self.__class__.__name__}
        if package_version:
            dict_['package_version'] = package_version
        return dict_

    def __getstate__(self):
        dict_ = {k: v for k, v in self.__dict__.items()
                 if (k not in self._non_serializable_attributes)\
                     and (not k.startswith('_'))}
        return dict_

    def to_dict(self):
        """
        Generic to_dict method
        """
        dict_ = self.__getstate__()
        if hasattr(self, 'Dict'):
            # !!! This prevent us to call DessiaObject.to_dict()
            # from an inheriting object which implement a Dict method,
            # because of the infinite recursion it creates.
            # TODO Change Dict methods to to_dict everywhere
            deprecation_warning(name='Dict', object_type='Function',
                                use_instead='to_dict')
            serialized_dict = self.Dict()
        else:
            # Default to dict
            serialized_dict = self.base_dict()
            serialized_dict.update(serialize_dict(dict_))
        # serialized_dict['hash'] = self.__hash__()
        return serialized_dict

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Generic dict_to_object method
        """
        if hasattr(cls, 'DictToObject'):
            deprecation_warning(name='DictToObject', object_type='Function',
                                use_instead='dict_to_object')
            return cls.DictToObject(dict_)

        if cls is not DessiaObject:
            obj = dict_to_object(dict_, cls)
            return obj
        elif 'object_class' in dict_:
            obj = dict_to_object(dict_)
            return obj
        else:
            # Using default
            # TODO: use jsonschema
            raise NotImplementedError('No object_class in dict')

    @classmethod
    def base_jsonschema(cls):
        jsonschema = deepcopy(JSONSCHEMA_HEADER)
        jsonschema['properties']['name'] = {
            'type': 'string',
            "title": "Object Name",
            "description": "Object name",
            "editable": True,
            "default_value": "Object Name"
        }
        return jsonschema

    @classmethod
    def jsonschema(cls):
        if hasattr(cls, '_jsonschema'):
            _jsonschema = cls._jsonschema
            return _jsonschema

        # Get __init__ method and its annotations
        init = cls.__init__
        if cls._init_variables is None:
            annotations = init.__annotations__
        else:
            annotations = cls._init_variables

        # Get ordered variables
        if cls._ordered_attributes:
            ordered_attributes = cls._ordered_attributes
        else:
            ordered_attributes = list(annotations.keys())

        unordered_count = 0

        # Initialize jsonschema
        _jsonschema = deepcopy(JSONSCHEMA_HEADER)

        required_arguments, default_arguments = inspect_arguments(method=init,
                                                                  merge=False)
        _jsonschema['required'] = required_arguments

        # Set jsonschema
        for annotation in annotations.items():
            name = annotation[0]
            if name in ordered_attributes:
                order = ordered_attributes.index(name)
            else:
                order = len(ordered_attributes) + unordered_count
                unordered_count += 1
            if name in cls._titled_attributes:
                title = cls._titled_attributes[name]
            else:
                title = None

            if name != 'return':
                editable = name not in cls._non_editable_attributes
                annotation_type = type_from_annotation(annotation[1], cls)
                annotation = (annotation[0], annotation_type)
                jss_elt = jsonschema_from_annotation(annotation=annotation,
                                                     jsonschema_element={},
                                                     order=order,
                                                     editable=editable,
                                                     title=title)
                _jsonschema['properties'].update(jss_elt)
                if name in default_arguments.keys():
                    default = set_default_value(_jsonschema['properties'],
                                                name,
                                                default_arguments[name])
                    _jsonschema['properties'].update(default)
                _jsonschema['classes'] = [cls.__module__ + '.' + cls.__name__]
                _jsonschema['whitelist_attributes'] = cls._whitelist_attributes
        return _jsonschema

    @property
    def _method_jsonschemas(self):
        """
        Generates dynamic jsonschemas for methods of class
        """
        jsonschemas = {}
        class_ = self.__class__

        # TOCHECK Backward compatibility. Will need to be changed
        if hasattr(class_, '_dessia_methods'):
            allowed_methods = class_._dessia_methods
        else:
            allowed_methods = class_._allowed_methods

        valid_method_names = [m for m in dir(class_)
                              if not m.startswith('_')
                              and m in allowed_methods]

        for method_name in valid_method_names:
            method = getattr(class_, method_name)

            if not isinstance(method, property):
                required_args, default_args = inspect_arguments(method=method,
                                                                merge=False)

                if method.__annotations__:
                    jsonschemas[method_name] = deepcopy(JSONSCHEMA_HEADER)
                    jsonschemas[method_name]['required'] = []
                    for i, annotation in enumerate(method.__annotations__.items()):  # TOCHECK Not actually ordered
                        argname = annotation[0]
                        if argname not in _FORBIDDEN_ARGNAMES:
                            if argname in required_args:
                                jsonschemas[method_name]['required'].append(
                                    str(i))
                            jsonschema_element = \
                                jsonschema_from_annotation(annotation, {}, i)[
                                    argname]

                            jsonschemas[method_name]['properties'][
                                str(i)] = jsonschema_element
                            if argname in default_args.keys():
                                default = set_default_value(
                                    jsonschemas[method_name]['properties'],
                                    str(i),
                                    default_args[argname])
                                jsonschemas[method_name]['properties'].update(
                                    default)
        return jsonschemas

    def method_dict(self, method_name=None, method_jsonschema=None):
        if method_name is None and method_jsonschema is None:
            msg = 'No method name not jsonschema provided'
            raise NotImplementedError(msg)

        if method_name is not None and method_jsonschema is None:
            method_jsonschema = self._method_jsonschemas[method_name]

        dict_ = default_dict(method_jsonschema)
        return dict_

    def dict_to_arguments(self, dict_, method):
        method_object = getattr(self, method)
        args_specs = inspect.getfullargspec(method_object)
        allowed_args = args_specs.args[1:]

        arguments = {}
        for i, arg in enumerate(allowed_args):
            if str(i) in dict_:
                arg_specs = args_specs.annotations[arg]
                value = dict_[str(i)]
                try:
                    deserialized_value = deserialize_argument(arg_specs, value)
                except TypeError:
                    msg = 'Error in deserialisation of value: '
                    msg += '{} of expected type {}'.format(value, arg_specs)
                    raise TypeError(msg)
                arguments[arg] = deserialized_value
        return arguments

    def save_to_file(self, filepath, indent=0):
        with open(filepath + '.json', 'w') as file:
            json.dump(self.to_dict(), file, indent=indent)

    @classmethod
    def load_from_file(cls, filepath):
        if isinstance(filepath, str):
            with open(filepath, 'r') as file:
                dict_ = json.load(file)
        else:
            dict_ = json.loads(filepath.read().decode('utf-8'))
        return cls.dict_to_object(dict_)

    def is_valid(self):
        return True

    def copy(self, deep=True):
        if deep:
            return self.__deepcopy__()
        else:
            return self.__copy__()

    def __copy__(self):
        """
        Generic copy use inits of objects
        """
        class_argspec = inspect.getfullargspec(self.__class__)
        dict_ = {}
        for arg in class_argspec.args:
            if arg != 'self':
                value = self.__dict__[arg]

                if hasattr(value, '__copy__'):
                    dict_[arg] = value.__copy__()
                else:
                    dict_[arg] = value
        return self.__class__(**dict_)

    def __deepcopy__(self, memo=None):
        """
        Generic deep copy use inits of objects
        """
        class_argspec = inspect.getfullargspec(self.__class__)
        if memo is None:
            memo = {}
        dict_ = {}
        for arg in class_argspec.args:
            if arg != 'self':
                dict_[arg] = deepcopy_value(getattr(self, arg), memo=memo)
        return self.__class__(**dict_)

    def volmdlr_volume_model(self, frame=None):
        if hasattr(self, 'volmdlr_primitives'):
            import volmdlr as vm  # !!! Avoid circular imports, is this OK ?
            if hasattr(self, 'volmdlr_primitives_step_frames'):
                return vm.MovingVolumeModel(self.volmdlr_primitives(),
                                            self.volmdlr_primitives_step_frames())
            else:
                if frame is None:
                    frame = vm.OXYZ
            try:
                return vm.core.VolumeModel(self.volmdlr_primitives(frame=frame))
            except TypeError:
                return vm.core.VolumeModel(self.volmdlr_primitives())
        msg = 'Object of type {} does not implement volmdlr_primitives'.format(self.__class__.__name__)
        raise NotImplementedError(msg)


    def cad_export(self,
                   fcstd_filepath=None,
                   istep=0,
                   python_path='python3',
                   freecad_lib_path='/usr/lib/freecad/lib',
                   export_types=['fcstd']):
        """
        Generic CAD export method
        """
        if fcstd_filepath is None:
            fcstd_filepath = 'An unnamed {}'.format(self.__class__.__name__)

        if hasattr(self, 'volmdlr_primitives'):
            model = self.volmdlr_volume_model()
            if model.__class__.__name__ == 'MovingVolumeModel':
                model = model.step_volume_model(istep)
            model.freecad_export(fcstd_filepath, python_path=python_path,
                                freecad_lib_path=freecad_lib_path, export_types=export_types)
        else:
            raise NotImplementedError

    def plot(self):
        """

        """
        if hasattr(self, 'plot_data'):
            import plot_data
            for data in self.plot_data():
                plot_data.plot_canvas(plot_data_object=data,
                                      canvas_id='canvas',
                                      debug_mode=False)

    def mpl_plot(self):
        axs = []
        if hasattr(self, 'plot_data'):
            for data in self.plot_data():
                if hasattr(data, 'mpl_plot'):
                    ax = data.mpl_plot()
                    axs.append(ax)
        return axs


    def babylonjs(self, use_cdn=True, debug=False):
        self.volmdlr_volume_model().babylonjs(use_cdn=use_cdn, debug=debug)

    def _display_angular(self):
        displays = []
        if hasattr(self, 'babylon_data'):
            displays.append({'angular_component': 'cad_viewer',
                            'data': self.babylon_data()})
        elif hasattr(self, 'volmdlr_primitives')\
                or (self.__class__.volmdlr_volume_model
                    is not DessiaObject.volmdlr_volume_model):
            model = self.volmdlr_volume_model()
            displays.append({'angular_component': 'cad_viewer',
                            'data': model.babylon_data()})
        if hasattr(self, 'plot_data'):
            plot_data = self.plot_data()
            if is_sequence(plot_data):
                for plot in plot_data:
                    displays.append({'angular_component': 'plot_data',
                                    'data': plot.to_dict()})
            else:
                plot = self.plot_data()
                displays.append({'angular_component': 'plot_data',
                                'data': plot.to_dict()})
        if hasattr(self, 'to_markdown'):
            displays.append({'angular_component': 'markdown',
                            'data': self.to_markdown()})
        return displays


class Parameter(DessiaObject):
    def __init__(self, lower_bound, upper_bound, periodicity=None, name=''):
        DessiaObject.__init__(self, name=name,
                              lower_bound=lower_bound,
                              upper_bound=upper_bound,
                              periodicity=periodicity)

    def random_value(self):
        return random.uniform(self.lower_bound, self.upper_bound)

    def are_values_equal(self, value1, value2, tol=1e-2):
        if self.periodicity is not None:
            value1 = value1 % self.periodicity
            value2 = value2 % self.periodicity

        return math.isclose(value1, value2, abs_tol=tol)

    def normalize(self, value):
        normalized_value = (value - self.lower_bound) / (self.upper_bound - self.lower_bound)
        return normalized_value

    def original_value(self, normalized_value):
        value = normalized_value * (self.upper_bound - self.lower_bound) + self.lower_bound
        return value

    def optimizer_bounds(self):
        if self.periodicity is not None:
            return (self.lower_bound - 0.5 * self.periodicity,
                    self.upper_bound + 0.5 * self.periodicity)


class ParameterSet(DessiaObject):
    def __init__(self, values, name=''):
        self.values = values

        DessiaObject.__init__(self, name=name)

    @property
    def parameters(self):
        parameters = [Parameter(min(v), max(v), name=k)
                      for k, v in self.values.items()]
        return parameters

    @property
    def means(self):
        means = {k: sum(v)/len(v) for k, v in self.values.items()}
        return means


class Filter(TypedDict):
    attribute: str
    operator: str
    bound: float


class Evolution(DessiaObject):
    """
    Defines a generic evolution

    :param evolution: float list
    :type evolution: list
    """
    _non_data_eq_attributes = ['name']
    _non_data_hash_attributes = ['name']
    _generic_eq = True

    def __init__(self, evolution: List[float] = None, name: str = ''):
        if evolution is None:
            evolution = []
        self.evolution = evolution

        DessiaObject.__init__(self, name=name)

    def _display_angular(self):
        displays = [{'angular_component': 'app-evolution1d',
                     'table_show': False,
                     'evolution': [self.evolution],
                     'label_y': ['evolution']}]
        return displays

    def update(self, evolution):
        """
        Update the evolution list
        """
        self.evolution = evolution


class CombinationEvolution(DessiaObject):
    _non_data_eq_attributes = ['name']
    _non_data_hash_attributes = ['name']
    _generic_eq = True

    def __init__(self, evolution1: List[Evolution], evolution2: List[Evolution],
                 title1: str = 'x', title2: str = 'y', name: str = ''):

        self.evolution1 = evolution1
        self.evolution2 = evolution2

        self.x_, self.y_ = self.genere_xy()

        self.title1 = title1
        self.title2 = title2

        DessiaObject.__init__(self, name=name)

    def _display_angular(self):
        displays = [{'angular_component': 'app-evolution2d-combination-evolution',
                     'table_show': False,
                     'evolution_x': [self.x_], 'label_x': ['title1'],
                     'evolution_y': [self.y_], 'label_y': ['title2']}]
        return displays

    def update(self, evol1, evol2):
        """
        Update the CombinationEvolution object

        :param evol1: list
        :param evol2: list
        """
        for evolution, ev1 in zip(self.evolution1, evol1):
            evolution.update(ev1)
        for evolution, ev2 in zip(self.evolution2, evol2):
            evolution.update(ev2)
        self.x_, self.y_ = self.genere_xy()

    def genere_xy(self):
        x, y = [], []
        for evol in self.evolution1:
            x = x + evol.evolution
        for evol in self.evolution2:
            y = y + evol.evolution
        return x, y


def number2factor(number):
    """
    Temporary function : Add to some tools package
    Finds all the ways to combine elements
    """
    factor_range = range(1, int(number ** 0.5) + 1)

    if number:
        factors = list(set(reduce(list.__add__, ([i, number // i]
                                                 for i in factor_range
                                                 if number % i == 0))))

        grids = [(factor_x, int(number / factor_x))
                 for factor_x in factors
                 if (number / factor_x).is_integer()]
    else:
        grids = []
    return grids


def number3factor(number, complete=True):
    """
    Temporary function : Add to some tools package
    Finds all the ways to combine elements
    """
    factor_range = range(1, int(number ** 0.5) + 1)

    if number:
        factors = list(set(reduce(list.__add__, ([i, number // i]
                                                 for i in factor_range
                                                 if number % i == 0))))
        if not complete:
            grids = get_incomplete_factors(number, factors)

        else:
            grids = [(factor_x, factor_y, int(number / (factor_x * factor_y)))
                     for factor_x in factors
                     for factor_y in factors
                     if (number / (factor_x * factor_y)).is_integer()]
        return grids
    return []


def get_incomplete_factors(number, factors):
    """
    TODO
    """
    grids = []
    sets = []
    for factor_x in factors:
        for factor_y in factors:
            value = number / (factor_x * factor_y)
            if value.is_integer():
                grid = (factor_x, factor_y, int(value))
                if set(grid) not in sets:
                    sets.append(set(grid))
                    grids.append(grid)
    return grids


def dict_merge(old_dct, merge_dct, add_keys=True, extend_lists=True):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    This version will return a copy of the dictionary and leave the original
    arguments untouched.

    The optional argument ``add_keys``, determines whether keys which are
    present in ``merge_dct`` but not ``dct`` should be included in the
    new dict.

    Args:
        old_dct (dict) onto which the merge is executed
        merge_dct (dict): dct merged into dct
        add_keys (bool): whether to add new keys
        extend_lists (bool) : wether to extend lists if keys are updated
                              and value is a list

    Returns:
        dict: updated dict
    """
    dct = deepcopy(old_dct)
    if not add_keys:
        merge_dct = {k: merge_dct[k]
                     for k in set(dct).intersection(set(merge_dct))}

    for key, value in merge_dct.items():
        if isinstance(dct.get(key), dict)\
                and isinstance(value, collections.Mapping):
            dct[key] = dict_merge(dct[key],
                                  merge_dct[key],
                                  add_keys=add_keys,
                                  extend_lists=extend_lists)
        elif isinstance(dct.get(key), list) and extend_lists:
            dct[key].extend(value)
        else:
            dct[key] = value

    return dct


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def stringify_dict_keys(obj):
    if isinstance(obj, (list, tuple)):
        new_obj = []
        for elt in obj:
            new_obj.append(stringify_dict_keys(elt))

    elif isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_obj[str(key)] = stringify_dict_keys(value)
    else:
        return obj
    return new_obj


def serialize_dict(dict_):
    serialized_dict = {}
    for key, value in dict_.items():
        if hasattr(value, 'to_dict'):
            serialized_value = value.to_dict()
        elif isinstance(value, dict):
            serialized_value = serialize_dict(value)
        elif isinstance(value, (list, tuple)):
            serialized_value = serialize_sequence(value)
        else:
            if not is_jsonable(value):
                msg = 'Attribute {} of value {} is not json serializable'
                raise SerializationError(msg.format(key, value))
            serialized_value = value
        serialized_dict[key] = serialized_value
    return serialized_dict


def serialize_sequence(seq):
    serialized_sequence = []
    for value in seq:
        if hasattr(value, 'to_dict'):
            serialized_sequence.append(value.to_dict())
        elif isinstance(value, dict):
            serialized_sequence.append(serialize_dict(value))
        elif isinstance(value, (list, tuple)):
            serialized_sequence.append(serialize_sequence(value))
        else:
            serialized_sequence.append(value)
    return serialized_sequence


def get_python_class_from_class_name(full_class_name):
    module_name, class_name = full_class_name.rsplit('.', 1)
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def dict_to_object(dict_, class_=None):
    working_dict = dict_.copy()
    if class_ is None and 'object_class' in working_dict:
        class_ = get_python_class_from_class_name(working_dict['object_class'])

    if class_ is not None:
        if hasattr(class_, 'dict_to_object') \
                and (class_.dict_to_object.__func__
                     is not DessiaObject.dict_to_object.__func__):
            obj = class_.dict_to_object(dict_)
            return obj
        if class_._init_variables is None:
            class_argspec = inspect.getfullargspec(class_)
            init_dict = {k: v for k, v in working_dict.items()
                         if k in class_argspec.args}
        else:
            init_dict = {k: v for k, v in working_dict.items()
                         if k in class_._init_variables}
        # TOCHECK Class method to generate init_dict ??
    else:
        init_dict = working_dict

    subobjects = {}
    for key, value in init_dict.items():
        if isinstance(value, dict):
            subobjects[key] = dict_to_object(value)
        elif isinstance(value, (list, tuple)):
            subobjects[key] = sequence_to_objects(value)
        else:
            subobjects[key] = value

    if class_ is not None:
        obj = class_(**subobjects)
    else:
        obj = subobjects
    return obj


def list_hash(list_):
    hash_ = 0
    for element in list_:
        if isinstance(element, list):
            hash_ += list_hash(element)
        elif isinstance(element, dict):
            hash_ += dict_hash(element)
        elif isinstance(element, str):
            hash_ += sum([ord(e) for e in element])
        else:
            hash_ += hash(element)
    return hash_


def dict_hash(dict_):
    hash_ = 0
    for key, value in dict_.items():
        if isinstance(value, list):
            hash_ += list_hash(value)
        elif isinstance(value, dict):
            hash_ += dict_hash(value)
        else:
            hash_ += hash(key) + hash(value)
    return hash_


def sequence_to_objects(sequence):
    # TODO: rename to deserialize sequence? Or is this a duplicate ?
    deserialized_sequence = []
    for element in sequence:
        if isinstance(element, dict):
            deserialized_sequence.append(dict_to_object(element))
        elif isinstance(element, (list, tuple)):
            deserialized_sequence.append(sequence_to_objects(element))
        else:
            deserialized_sequence.append(element)
    return deserialized_sequence


def getdeepattr(obj, attr):
    return reduce(getattr, [obj] + attr.split('.'))


def full_classname(object_):
    return object_.__class__.__module__ + '.' + object_.__class__.__name__


def serialization_test(obj):
    # TODO: debug infinite recursion? Should we remove thhis ?
    d = obj.to_dict()
    obj2 = obj.dict_to_object(d)
    if obj != obj2:
        msg = 'Object in no more equal to himself '
        msg += 'after serialization/deserialization!'
        raise ModelError(msg)


def deepcopy_value(value, memo):
    # Escaping unhashable types (list) that would be handled after
    try:
        if value in memo:
            return memo[value]
    except TypeError:
        pass

    if isinstance(value, type):  # For class
        return value
    elif hasattr(value, '__deepcopy__'):
        try:
            copied_value = value.__deepcopy__(memo)
        except TypeError:
            copied_value = value.__deepcopy__()
        memo[value] = copied_value
        return copied_value
    else:
        if isinstance(value, list):
            copied_list = []
            for v in value:
                cv = deepcopy_value(v, memo=memo)
                try:
                    memo[v] = cv
                except TypeError:
                    pass
                copied_list.append(cv)
            return copied_list
        elif isinstance(value, dict):
            copied_dict = {}
            for k, v in value.items():
                copied_k = deepcopy_value(k, memo=memo)
                copied_v = deepcopy_value(v, memo=memo)
                try:
                    memo[k] = copied_k
                except TypeError:
                    pass
                try:
                    memo[v] = copied_v
                except TypeError:
                    pass
                copied_dict[copied_k] = copied_v
            return copied_dict
        else:
            new_value = copy.deepcopy(value, memo=memo)
            memo[value] = new_value
            return new_value


def serialize_typing(typing_):
    if hasattr(typing_, '__origin__') and typing_.__origin__ == list:
        arg = typing_.__args__[0]
        if arg.__module__ == 'builtins':
            full_argname = '__builtins__.' + arg.__name__
        else:
            full_argname = arg.__module__ + '.' + arg.__name__
        return 'List[' + full_argname + ']'
    if isinstance(typing_, type):
        return typing_.__module__ + '.' + typing_.__name__
    raise NotImplementedError('{} of type {}'.format(typing_, type(typing_)))


def deserialize_typing(serialized_typing):
    if isinstance(serialized_typing, str):
        if serialized_typing == 'float'\
                or serialized_typing == 'builtins.float':
            return float

        splitted_type = serialized_typing.split('[')
        if splitted_type[0] == 'List':
            full_argname = splitted_type[1].split(']')[0]
            splitted_argname = full_argname.rsplit('.', 1)
            if splitted_argname[0] != '__builtins__':
                exec('import ' + splitted_argname[0])
                type_ = eval(full_argname)
            else:
                type_ = eval(splitted_argname[1])
            return List[type_]
    raise NotImplementedError('{}'.format(serialized_typing))


def serialize(deserialized_element):
    if isinstance(deserialized_element, DessiaObject):
        serialized = deserialized_element.to_dict()
    elif isinstance(deserialized_element, dict):
        serialized = serialize_dict(deserialized_element)
    elif is_sequence(deserialized_element):
        serialized = serialize_sequence(deserialized_element)
    else:
        serialized = deserialized_element
    return serialized


def deserialize(serialized_element):
    if isinstance(serialized_element, dict):
        element = dict_to_object(serialized_element)
    elif is_sequence(serialized_element):
        element = sequence_to_objects(serialized_element)
    else:
        element = serialized_element
    return element


def enhanced_deep_attr(obj, sequence):
    """
    Get deep attribute where Objects, Dicts and Lists
    can be found in recursion.

    :param obj: Parent object in which recursively find attribute
                represented by sequence
    :param sequence: List of strings and integers that represents
                     path to deep attribute.
    :return: Value of deep attribute
    """
    if isinstance(sequence, str):
        # Sequence is a string and not a sequence of deep attributes
        if '/' in sequence:
            # Is deep attribute reference
            sequence = deepattr_to_sequence(sequence)
            return enhanced_deep_attr(obj=obj, sequence=sequence)
        # Is direct attribute
        return enhanced_get_attr(obj=obj, attr=sequence)

    # Get direct attrivute
    subobj = enhanced_get_attr(obj=obj, attr=sequence[0])
    if len(sequence) > 1:
        # Recursively get deep attributes
        subobj = enhanced_deep_attr(obj=subobj, sequence=sequence[1:])
    return subobj


def enhanced_get_attr(obj, attr):
    """
    Safely get attribute in obj.
    Obj can be of Object, Dict, or List type

    :param obj: Parent object in which find given attribute
    :param attr: String or integer that represents
                      name or index of attribute
    :return: Value of attribute
    """
    try:
        return getattr(obj, attr)
    except (TypeError, AttributeError):
        try:
            return obj[attr]
        except TypeError:
            classname = obj.__class__.__name__
            msg = "'{}' object has no attribute '{}'.".format(classname, attr)
            track = tb.format_exc()
            raise DeepAttributeError(message=msg, traceback_=track)


def concatenate_attributes(prefix, suffix, type_: str = 'str'):
    wrong_prefix_format = 'Attribute prefix is wrongly formatted.'
    wrong_prefix_format += 'Is of type {}. Should be str or list'
    if type_ == 'str':
        if isinstance(prefix, str):
            return prefix + '/' + str(suffix)
        elif is_sequence(prefix):
            return sequence_to_deepattr(prefix) + '/' + str(suffix)
        else:
            raise TypeError(wrong_prefix_format.format(type(prefix)))
    elif type_ == 'sequence':
        if isinstance(prefix, str):
            return [prefix, suffix]
        elif is_sequence(prefix):
            return prefix + [suffix]
        else:
            raise TypeError(wrong_prefix_format.format(type(prefix)))
    else:
        wrong_concat_type = 'Type {} for concatenation is not supported.'
        wrong_concat_type += 'Should be "str" or "sequence"'
        raise ValueError(wrong_concat_type.format(type_))


def deepattr_to_sequence(deepattr: str):
    sequence = deepattr.split('/')
    healed_sequence = []
    for i, attribute in enumerate(sequence):
        try:
            healed_sequence.append(int(attribute))
        except ValueError:
            healed_sequence.append(attribute)
    return healed_sequence


def sequence_to_deepattr(sequence):
    healed_sequence = [str(attr) if isinstance(attr, int) else attr
                       for attr in sequence]
    return '/'.join(healed_sequence)


def is_bounded(filter_: Filter, value: float):
    bounded = True
    operator = filter_['operator']
    bound = filter_['bound']

    if operator == 'lte' and value > bound:
        bounded = False
    if operator == 'gte' and value < bound:
        bounded = False

    if operator == 'lt' and value >= bound:
        bounded = False
    if operator == 'gt' and value <= bound:
        bounded = False

    if operator == 'eq' and value != bound:
        bounded = False
    return bounded


def is_sequence(obj):
    """
    :param obj: Object to check
    :return: bool. True if object is a sequence but not a string.
                   False otherwise
    """
    return isinstance(obj, collections.abc.Sequence)\
        and not isinstance(obj, str)


def type_from_annotation(type_, module):
    """
    Clean up a proposed type if there are stringified
    """
    if isinstance(type_, str):
        # Evaluating types
        if type_ in TYPES_FROM_STRING:
            type_ = TYPES_FROM_STRING[type_]
        else:
            # Evaluating
            type_ = getattr(import_module(module), type_)
    return type_


def jsonschema_from_annotation(annotation, jsonschema_element,
                               order, editable=None, title=None):
    key, value = annotation
    if isinstance(value, str):
        raise ValueError

    if title is None:
        title = prettyname(key)
    if editable is None:
        editable = key not in ['return']

    if value in TYPING_EQUIVALENCES.keys():
        # Python Built-in type
        jsonschema_element[key] = {'type': TYPING_EQUIVALENCES[value],
                                   'title': title, 'editable': editable,
                                   'order': order}
    elif hasattr(value, '__origin__') and value.__origin__ == Union:
        # Types union
        classnames = [a.__module__ + '.' + a.__name__ for a in value.__args__]
        jsonschema_element[key] = {'type': 'object', 'classes': classnames,
                                   'title': title, 'editable': editable,
                                   'order': order}
    elif hasattr(value, '__origin__') and value.__origin__ == list:
        # Homogenous sequences
        jsonschema_element[key] = jsonschema_sequence_recursion(
            value=value, order=order, title=title, editable=editable
        )
    elif hasattr(value, '__origin__') and value.__origin__ == tuple:
        # Heterogenous sequences (tuples)
        items = []
        for type_ in value.__args__:
            items.append({'type': TYPING_EQUIVALENCES[type_]})
        jsonschema_element[key] = {'additionalItems': False,
                                   'type': 'array', 'items': items}
    elif hasattr(value, '__origin__') and value.__origin__ == dict:
        # Dynamially created dict structure
        key_type, value_type = value.__args__
        if key_type != str:
            # !!! Should we support other types ? Numeric ?
            raise NotImplementedError('Non strings keys not supported')
        jsonschema_element[key] = {
            'type': 'object',
            'order': order,
            'editable': editable,
            'title': title,
            'patternProperties': {
                '.*': {
                    'type': TYPING_EQUIVALENCES[value_type]
                }
            }
        }
    elif hasattr(value, '__origin__') and value.__origin__ == dt.Subclass:
        # Several possible classes that are subclass of another one
        class_ = value.__args__[0]
        classname = class_.__module__ + '.' + class_.__name__
        jsonschema_element[key] = {'type': 'object', 'subclass_of': classname,
                                   'title': title, 'editable': editable,
                                   'order': order}
    else:
        classname = value.__module__ + '.' + value.__name__
        if issubclass(value, DessiaObject):
            # Dessia custom classes
            jsonschema_element[key] = {'type': 'object'}
        else:
            # Statically created dict structure
            jsonschema_element[key] = static_dict_jsonschema(value)
        jsonschema_element[key].update({'title': title, 'order': order,
                                        'editable': editable,
                                        'classes': [classname]})
    return jsonschema_element


def jsonschema_sequence_recursion(value, order: int, title: str = None,
                                  editable: bool = False):
    if title is None:
        title = 'Items'
    jsonschema_element = {'type': 'array', 'order': order,
                          'editable': editable, 'title': title}

    items_type = value.__args__[0]
    if hasattr(items_type, '__origin__') and items_type.__origin__ == list:
        jss = jsonschema_sequence_recursion(value=items_type, order=0,
                                            title=title, editable=editable)
        jsonschema_element['items'] = jss
    else:
        annotation = ('items', items_type)
        jss = jsonschema_from_annotation(annotation=annotation,
                                         jsonschema_element=jsonschema_element,
                                         order=0, title=title)
        jsonschema_element.update(jss)
    return jsonschema_element


def prettyname(namestr):
    pretty_name = ''
    if namestr:
        strings = namestr.split('_')
        for i, string in enumerate(strings):
            if len(string) > 1:
                pretty_name += string[0].upper() + string[1:]
            else:
                pretty_name += string
            if i < len(strings) - 1:
                pretty_name += ' '
    return pretty_name


def static_dict_jsonschema(typed_dict, title=None):
    jsonschema_element = deepcopy(JSONSCHEMA_HEADER)
    jss_properties = jsonschema_element['properties']

    # Every value is required in a StaticDict
    jsonschema_element['required'] = list(typed_dict.__annotations__.keys())

    # TOCHECK : Not actually ordered !
    for i, ann in enumerate(typed_dict.__annotations__.items()):
        jss = jsonschema_from_annotation(annotation=ann,
                                         jsonschema_element=jss_properties,
                                         order=i, title=title)
        jss_properties.update(jss)
    return jsonschema_element


def set_default_value(jsonschema_element, key, default_value):
    if isinstance(default_value, tuple(TYPING_EQUIVALENCES.keys())) \
            or default_value is None:
        jsonschema_element[key]['default_value'] = default_value
    elif is_sequence(default_value):
        # TODO : Tuple should be considered OK for default_value
        msg = 'Object {} of type {} is not supported as default value'
        type_ = type(default_value)
        raise NotImplementedError(msg.format(default_value, type_))
    else:
        object_dict = default_value.to_dict()
        jsonschema_element[key]['default_value'] = object_dict
    return jsonschema_element


def inspect_arguments(method, merge=False):
    # Find default value and required arguments of class construction
    args_specs = inspect.getfullargspec(method)
    nargs = len(args_specs.args) - 1

    if args_specs.defaults is not None:
        ndefault_args = len(args_specs.defaults)
    else:
        ndefault_args = 0

    default_arguments = {}
    arguments = []
    for iargument, argument in enumerate(args_specs.args[1:]):
        if argument not in _FORBIDDEN_ARGNAMES:
            if iargument >= nargs - ndefault_args:
                default_value = args_specs.defaults[ndefault_args - nargs
                                                    + iargument]
                if merge:
                    arguments.append((argument, default_value))
                else:
                    default_arguments[argument] = default_value
            else:
                arguments.append(argument)
    return arguments, default_arguments


def deserialize_argument(type_, argument):
    if hasattr(type_, '__origin__') and type_.__origin__ == Union:
        # Type union
        classes = list(type_.__args__)
        instantiated = False
        while instantiated is False:
            # Find the last class in the hierarchy
            hierarchy_lengths = [len(cls.mro()) for cls in classes]
            children_class_index = hierarchy_lengths.index(
                max(hierarchy_lengths))
            children_class = classes[children_class_index]
            try:
                # Try to deserialize
                # Throws KeyError if we try to put wrong dict into
                # dict_to_object. This means we try to instantiate
                # a children class with a parent dict_to_object
                deserialized_argument = children_class.dict_to_object(argument)

                # If it succeeds we have the right
                # class and instantiated object
                instantiated = True
            except KeyError:
                # This is not the right class, we should go see the parent
                classes.remove(children_class)
    elif hasattr(type_, '__origin__') and type_.__origin__ == list:
        # Homogenous sequences (lists)
        sequence_subtype = type_.__args__[0]
        deserialized_argument = [deserialize_argument(sequence_subtype, arg)
                                 for arg in argument]
    elif hasattr(type_, '__origin__') and type_.__origin__ == 'Tuple':
        # Heterogenous sequences (tuples)
        deserialized_argument = tuple([deserialize_argument(t, arg)
                                       for t, arg in zip(type_.__args__,
                                                         argument)])
    elif hasattr(type_, '__origin__') and type_.__origin__ == 'Dict':
        # Dynamic dict
        deserialized_argument = argument
    else:
        if type_ in TYPING_EQUIVALENCES.keys():
            if isinstance(argument, type_):
                deserialized_argument = argument
            else:
                if isinstance(argument, int) and type_ == float:
                    # Explicit conversion in this case
                    deserialized_argument = float(argument)
                else:
                    msg = 'Given built-in type and argument are incompatible: '
                    msg += '{} and {} in {}'.format(type(argument),
                                                    type_, argument)
                    raise TypeError(msg)
        elif issubclass(type_, DessiaObject):
            # Custom classes
            deserialized_argument = type_.dict_to_object(argument)
        else:
            # Static Dict
            deserialized_argument = argument
    return deserialized_argument


# TODO recursive_type and recursive_type functions look weird
def recursive_type(obj):
    if isinstance(obj, tuple(list(TYPING_EQUIVALENCES.keys()) + [dict])):
        type_ = TYPES_STRINGS[type(obj)]
    elif isinstance(obj, DessiaObject):
        type_ = obj.__module__ + '.' + obj.__class__.__name__
    elif isinstance(obj, (list, tuple)):
        type_ = []
        for element in obj:
            type_.append(recursive_type(element))
    elif obj is None:
        type_ = None
    else:
        raise NotImplementedError(obj)
    return type_


def recursive_instantiation(type_, value):
    if type_ in TYPES_STRINGS.values():
        return eval(type_)(value)
    elif isinstance(type_, str):
        class_ = get_python_class_from_class_name(type_)
        if inspect.isclass(class_):
            return class_.dict_to_object(value)
        else:
            raise NotImplementedError
    elif isinstance(type_, (list, tuple)):
        return [recursive_instantiation(t, v) for t, v in zip(type_, value)]
    elif type_ is None:
        return value
    else:
        raise NotImplementedError(type_)


def choose_default(jsonschema):
    if jsonschema['type'] == 'object':
        default_value = default_dict(jsonschema)
    elif jsonschema['type'] == 'array':
        default_value = default_sequence(jsonschema)
    elif jsonschema['type'] == 'string':
        default_value = ''
    else:
        default_value = None
    return default_value


def default_dict(jsonschema):
    dict_ = {}
    if 'properties' in jsonschema:
        for property_, value in jsonschema['properties'].items():
            if property_ in jsonschema['required']:
                dict_[property_] = choose_default(value)
            else:
                default_value = value['default_value']
                if value['type'] == 'array':
                    dict_[property_] = []
                elif value['type'] == 'object' and default_value is None:
                    dict_[property_] = {}
                else:
                    dict_[property_] = default_value
    return dict_


def default_sequence(array_jsonschema):
    if 'minItems' in array_jsonschema and 'maxItems' in array_jsonschema \
            and array_jsonschema['minItems'] == array_jsonschema['maxItems']:
        number = array_jsonschema['minItems']
    elif 'minItems' in array_jsonschema:
        number = array_jsonschema['minItems']
    elif 'maxItems' in array_jsonschema:
        number = array_jsonschema['maxItems']
    else:
        number = 1

    if type(array_jsonschema['items']) == list:
        # Tuple jsonschema
        return [default_dict(v) for v in array_jsonschema['items']]

    elif array_jsonschema['items']['type'] == 'object':
        if 'classes' in array_jsonschema['items']:
            # TOCHECK classes[0]
            classname = array_jsonschema['items']['classes'][0]
            class_ = get_python_class_from_class_name(classname)
            if issubclass(class_, DessiaObject):
                if class_._standalone_in_db:  # Standalone object
                    return []
                # Embedded object
                default_subdict = default_dict(class_.jsonschema())
                return [default_subdict] * number
            # Static Dict
            dict_jsonschema = static_dict_jsonschema(class_)
            default_subdict = default_dict(dict_jsonschema)
            return [default_subdict] * number
        # Subclasses
        return [choose_default(array_jsonschema['items'])] * number
    return [choose_default(array_jsonschema['items'])] * number
