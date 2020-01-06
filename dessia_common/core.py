#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import math
import random
from functools import reduce
import collections
from copy import deepcopy
import inspect
import json
from typing import TypeVar, List
#from typing import List, Sequence, Iterable, TypeVar, Union

#from importlib import import_module

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

class DessiaObject(protected_module.DessiaObject if not _open_source else object):
    """
    Base abstract class for Dessia's object.
    Gathers generic methods and attributes
    """
    _standalone_in_db = False
    _non_serializable_attributes = []
    _non_eq_attributes = ['name']
    _non_hash_attributes = ['name']
    _generic_eq = False
    _init_variables = None

    def __init__(self, name:str='', **kwargs):
        implements_eq = (hasattr(self, '__eq__') and hasattr(self, '__hash__')
                         and self.__eq__ is not object.__eq__
                         and self.__hash__ is not object.__hash__)
        if self._standalone_in_db and not self._generic_eq and not implements_eq:
            raise ValueError('Standalone in database classes must define their custom __hash__ and __eq__')

        self.name = name
        for property_name, property_value in kwargs.items():
            setattr(self, property_name, property_value)

    def __eq__(self, other_object):
        if not self._generic_eq:
            return object.__eq__(self, other_object)
        if self.__class__ != other_object.__class__\
        or self.__dict__.keys() != other_object.__dict__.keys():
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
                else:
                    hash_ += hash(value)
        return int(hash_ % 1e5)

    @property
    def full_classname(self):
        return full_classname(self)

    def base_dict(self):
        dict_ = {'name' : self.name}
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
        serialized_dict = serialize_dict(dict_)
        serialized_dict['object_class'] = self.__module__ + '.' + self.__class__.__name__

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
            print(dict_.keys())
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
        if type(filepath) is str:
            with open(filepath, 'r') as file:
                dict_ = json.load(file)
        else:
            dict_ = json.loads(filepath.read().decode('utf-8'))
        return cls.dict_to_object(dict_)

    def is_valid(self):
        return True

    def copy(self):
        class_argspec = inspect.getfullargspec(self.__class__)
        dict_ = {}
        for arg in class_argspec.args:
            if arg != 'self':
                value = self.__dict__[arg]

                if hasattr(value, 'Copy'):
                    # Backward compatibility
                    dict_[arg] = value.Copy()
                elif hasattr(value, 'copy'): # TODO : Check if not DessiaObject.copy
                    dict_[arg] = value.copy()
                else:
                    dict_[arg] = value
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
            return vm.VolumeModel(self.volmdlr_primitives(frame=frame))

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
        self.volmdlr_volume_model().BabylonShow(use_cdn=use_cdn, debug=debug)

    def _display_angular(self):
        display = []
        if hasattr(self, 'CADExport')\
        or hasattr(self, 'FreeCADExport')\
        or hasattr(self, 'volmdlr_primitives'):
            display.append({'angular_component': 'app-cad-viewer'})
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

def dict_to_object(dict_, class_=None):
    working_dict = dict_.copy()
    if class_ is None and 'object_class' in working_dict:
        object_class = working_dict['object_class']
        module = object_class.rsplit('.', 1)[0]
        exec('import ' + module)
        class_ = eval(object_class)

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

def sequence_to_objects(sequence):
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