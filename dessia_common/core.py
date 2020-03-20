#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import warnings
import math
import random
import copy
from functools import reduce
import collections
from copy import deepcopy
import inspect
import json
from typing import List
import traceback as tb

from importlib import import_module

try:
    _open_source = False
    import dessia_common.core_protected as protected_module
    from dessia_common.core_protected import inspect_arguments, recursive_instantiation,\
                                             recursive_type, JSONSCHEMA_HEADER,\
                                             jsonschema_from_annotation,prettyname,\
                                             set_default_value, TYPING_EQUIVALENCES,\
                                             deserialize_argument
except (ModuleNotFoundError, ImportError) as _:
    _open_source = True

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
    warnings.simplefilter('always', DeprecationWarning)
    msg = "\n\n{} {} is deprecated.\n".format(object_type, name)
    msg += "It will be removed in a future version.\n"
    if use_instead is not None:
        msg += "Use {} instead.\n".format(use_instead)
    warnings.warn(msg, DeprecationWarning)
    return msg

class DessiaObject(protected_module.DessiaObject if not _open_source else object):
    """
    Base abstract class for Dessia's object.
    Gathers generic methods and attributes
    """
    _standalone_in_db = False
    _non_serializable_attributes = []
    _non_editable_attributes = []
    _non_eq_attributes = ['name']
    _non_hash_attributes = ['name']
    _generic_eq = False
    _init_variables = None
    _export_formats = None
    _allowed_methods = []

    def __init__(self, name:str='', **kwargs):
        implements_eq = (hasattr(self, '__eq__') and hasattr(self, '__hash__')
                         and self.__class__.__eq__ is not object.__eq__
                         and self.__class__.__hash__ is not object.__hash__)
        if self._standalone_in_db and not self._generic_eq and not implements_eq:
            raise ValueError('Standalone in database classes must define their custom __hash__ and __eq__')

        self.name = name
        for property_name, property_value in kwargs.items():
            setattr(self, property_name, property_value)

    def __eq__(self, other_object):
        if not self._generic_eq:
            return object.__eq__(self, other_object)
        if full_classname(self) != full_classname(other_object)\
        or self.__dict__.keys() != other_object.__dict__.keys(): # TODO : Check this line. Keys not ordered and/or just need to test used keys
            return False

        dict_ = {k : v for k, v in self.__dict__.items()\
                 if k not in self._non_eq_attributes}
        other_dict = {k : v for k, v in other_object.__dict__.items()\
                      if k not in self._non_eq_attributes}

        for key, value in dict_.items():
            other_value = other_dict[key]
            if value != other_value:
                return False
        return True
    

    def __hash__(self):
        if not self._generic_eq:
            return object.__hash__(self)
        hash_ = 0
        for key, value in self.__dict__.items():
            if key not in set(self._non_eq_attributes + self._non_hash_attributes):
                if isinstance(value, list):
                    hash_ += list_hash(value)
                elif isinstance(value, dict):
                    hash_ += dict_hash(value)
                elif isinstance(value, str):
                    hash_ += sum([ord(v) for v in value])
                else:
                    hash_ += hash(value)
        return int(hash_ % 1e5)

    # def __getattribute__(self, name):
    #     if name in DEPRECATED_ATTRIBUTES:
    #         deprecation_warning(name, 'Attribute', DEPRECATED_ATTRIBUTES[name])
    #     return object.__getattribute__(self, name)

    # def __setattribute__(self, name, value):
    #     if name in DEPRECATED_ATTRIBUTES:
    #         deprecation_warning(name, 'Attribute', DEPRECATED_ATTRIBUTES[name])
    #     return object.__setattribute__(self, name, value)

    @property
    def full_classname(self):
        return full_classname(self)

    def base_dict(self):
        if hasattr(self.__class__.__module__, '__version__'):
            pkg_version = self.__class__.__module__.__version__
        else:
            pkg_version = None
        dict_ = {'name' : self.name,
                 'package_version' : pkg_version,
                 'object_class' : self.__module__ + '.' + self.__class__.__name__}
        return dict_

    def __getstate__(self):
        dict_ = {k : v for k, v in self.__dict__.items()\
                 if k not in self._non_serializable_attributes}
        return dict_

    def to_dict(self):
        """
        Generic to_dict method
        """
        dict_ = self.__getstate__()
        if hasattr(self, 'Dict'):
            # !!! This prevent us to call DessiaObject.to_dict() from an inheriting object
            # which implement a Dict method, because of the infinite recursion it creates.
            # TODO Change Dict methods to to_dict everywhere
            return self.Dict()

        # Default to dict
        serialized_dict = self.base_dict()
        serialized_dict.update(serialize_dict(dict_))

        return serialized_dict


    @classmethod
    def dict_to_object(cls, dict_):
        """
        Generic dict_to_object method
        """
        if hasattr(cls, 'DictToObject'):
            return cls.DictToObject(dict_)

        if cls is not DessiaObject:
            obj = dict_to_object(dict_, cls)
            return obj
        elif 'object_class' in dict_:
            obj = dict_to_object(dict_)
            return obj
        # Using default
        # TODO: use jsonschema
        return obj

    def save_to_file(self, filepath, indent = 0):
        with open(filepath+'.json', 'w') as file:
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
            import volmdlr as vm # !!! Avoid circular imports, is this OK ?
            if hasattr(self, 'volmdlr_primitives_step_frames'):
                return vm.MovingVolumeModel(self.volmdlr_primitives(),
                                            self.volmdlr_primitives_step_frames())
            else:
                if frame is None:
                    frame = vm.OXYZ
            try:
                return vm.VolumeModel(self.volmdlr_primitives(frame=frame))
            except TypeError:
                return vm.VolumeModel(self.volmdlr_primitives())
    
        raise NotImplementedError('object of type {} does not implement volmdlr_primitives'.format(self.__class__.__name__))

    def cad_export(self,
                   fcstd_filepath=None,
                   istep=0,
                   python_path='python',
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
            model.FreeCADExport(fcstd_filepath, python_path=python_path,
                                freecad_lib_path=freecad_lib_path, export_types=export_types)
        else:
            raise NotImplementedError

    def babylonjs(self, use_cdn=True, debug=False):
        self.volmdlr_volume_model().babylonjs(use_cdn=use_cdn, debug=debug)

    def _display_angular(self):
        display = []
        # if hasattr(self, 'CADExport')\
        # or hasattr(self, 'FreeCADExport'):
        #     display.append({'angular_component': 'app-cad-viewer'})

        if hasattr(self, 'babylon_data'):
            display.append({'angular_component' : 'app-step-viewer',
                            'data' : self.babylon_data()})
        elif hasattr(self, 'volmdlr_primitives')\
        or (self.__class__.volmdlr_volume_model is not DessiaObject.volmdlr_volume_model):
            model = self.volmdlr_volume_model()
            display.append({'angular_component' : 'app-step-viewer',
                            'data' : model.babylon_data()})

        if hasattr(self, 'plot_data'):
            display.append({'angular_component' : 'app-d3-plot-data',
                            'data' : self.plot_data()})
        return display

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
        normalized_value = (value - self.lower_bound)/(self.upper_bound - self.lower_bound)
        return normalized_value

    def original_value(self, normalized_value):
        value = normalized_value*(self.upper_bound - self.lower_bound) + self.lower_bound
        return value

    def optimizer_bounds(self):
        if self.periodicity is not None:
            return (self.lower_bound-0.5*self.periodicity,
                    self.upper_bound+0.5*self.periodicity)

class Evolution(DessiaObject):
    """
    Defines a generic evolution

    :param evolution: float list
    :type evolution: list
    """
    _non_eq_attributes = ['name']
    _non_hash_attributes = ['name']
    _generic_eq = True

    def __init__(self, evolution:List[float]=None, name:str=''):
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
    _non_eq_attributes = ['name']
    _non_hash_attributes = ['name']
    _generic_eq = True

    def __init__(self, evolution1:List[Evolution], evolution2:List[Evolution],
                 title1:str='x', title2:str='y', name:str=''):

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
    factor_range = range(1, int(number**0.5) + 1)

    if number:
        factors = list(set(reduce(list.__add__,
                                  ([i, number//i] for i in factor_range if number % i == 0))))

        grids = [(factor_x, int(number/factor_x))
                 for factor_x in factors
                 if (number/factor_x).is_integer()]
    else:
        grids = []
    return grids

def number3factor(number, complete=True):
    """
    Temporary function : Add to some tools package
    Finds all the ways to combine elements
    """
    factor_range = range(1, int(number**0.5) + 1)

    if number:
        factors = list(set(reduce(list.__add__,
                                  ([i, number//i] for i in factor_range if number % i == 0))))
        if not complete:
            grids = get_incomplete_factors(number, factors)

        else:
            grids = [(factor_x, factor_y, int(number/(factor_x*factor_y)))\
                     for factor_x in factors\
                     for factor_y in factors\
                     if (number/(factor_x*factor_y)).is_integer()]
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
            value = number/(factor_x*factor_y)
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
        extend_lists (bool) : wether to extend lists if keys are updated and value is a list

    Returns:
        dict: updated dict
    """
    dct = deepcopy(old_dct)
    if not add_keys:
        merge_dct = {k: merge_dct[k]\
                     for k in set(dct).intersection(set(merge_dct))}

    for key, value in merge_dct.items():
        if (isinstance(dct.get(key), dict) and isinstance(value, collections.Mapping)):
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
                raise SerializationError('Value {} {} is not json serializable'.format(key, value))
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

def get_python_class_from_class_name(class_name):
    #TODO: add protection because this is arbitratry code evaluation!
    module = class_name.rsplit('.', 1)[0]
    exec('import ' + module)
    return eval(class_name)

def dict_to_object(dict_, class_=None):
    working_dict = dict_.copy()
    if class_ is None and 'object_class' in working_dict:
        # object_class = working_dict['object_class']
        # module = object_class.rsplit('.', 1)[0]
        # exec('import ' + module)
        # class_ = eval(object_class)
        class_ = get_python_class_from_class_name(working_dict['object_class'])

    if class_ is not None:
        if hasattr(class_, 'dict_to_object')\
        and class_.dict_to_object.__func__ is not DessiaObject.dict_to_object.__func__:
            obj = class_.dict_to_object(dict_)
            return obj
        if class_._init_variables is None:
            class_argspec = inspect.getfullargspec(class_)
            init_dict = {k:v for k, v in working_dict.items() if k in class_argspec.args}
        else:
            init_dict = {k:v for k, v in working_dict.items() if k in class_._init_variables}
        # !!! Class method to generate init_dict ??
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

def sequence_to_objects(sequence):# TODO: rename to deserialize sequence?
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
    # TODO: debug infinite recursion?
    d = obj.to_dict()
    obj2 = obj.dict_to_object(d)
    if obj != obj2:
        raise ModelError('object in no more equal to himself after serialization/deserialization!')

def deepcopy_value(value, memo):
    # Escaping unhashable types (list) that would be handled after
    try:
        if value in memo:
            return memo[value]
    except TypeError:
        pass

    if isinstance(value, type):# For class
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
                memo[v] = cv
                copied_list.append(cv)
            return copied_list
        else:
            new_value = copy.deepcopy(value, memo=memo)
            memo[value] = new_value
            return new_value


def serialize_typing(typing_):
    if hasattr(typing_, '_name') and typing_._name == 'List':
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
        if serialized_typing == 'float' or serialized_typing == 'builtins.float':
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
    
def deserialize(serialized_element):
    if isinstance(serialized_element, dict):
        element = dict_to_object(serialized_element)
    elif isinstance(serialized_element, (list, tuple)):
        element = sequence_to_objects(serialized_element)    
    else:
        element = serialized_element
        
    return element
    
    
TYPES_FROM_STRING = {'unicode': str,
                     'str': str,
                     'float': float,
                     'int': int,
                     'bool': bool}

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