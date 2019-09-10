#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:17:30 2018

@author: Steven Masfaraud masfaraud@dessia.tech
"""

from functools import reduce
from copy import deepcopy
import collections

class Metadata:
    """
    Gathers object custom data
    """
    _standalone_in_db = False
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "powerpack.mechanical.MechModule Base Schema",
        "required": ["name"],
        "properties": {
            "name" : {
                "type" : "string",
                "order" : 0,
                "editable" : True,
                "examples" : ["Object name"],
                "description" : "Object name"
                },
#            "id_" : {
#                "type" : "string",
#                "order" : 1,
#                "editable" : False,
#                "Description" : "Object id"
#                }
            }
        }
    def __init__(self, name='', **kwargs):
        self.name = name
        if kwargs:
            self.add_data(**kwargs)
#        self.id_ = None
#        self.last_update = None

    def add_data(self, **kwargs):
        [self.__setattr__(keyword, value) for keyword, value in kwargs.items()]

    def del_data(self, attribute_name):
        self.__delattr__(attribute_name)

    def to_dict(self):
        # !!! Enable python object in metadata ?
        return self.__dict__

    @classmethod
    def dict_to_object(cls, d):
        kwargs = deepcopy(d)
        if 'name' in d:
            name = d['name']
            kwargs.pop('name')
        else:
            name = ''
        metadata = cls(name, **kwargs)
        return metadata

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
        merge_dct = {
            k: merge_dct[k]
            for k in set(dct).intersection(set(merge_dct))
        }

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


def stringify_dict_keys(d):
    if type(d) == list or type(d) == tuple:
        new_d = []
        for di in d:
            new_d.append(stringify_dict_keys(di))
        
    elif type(d) ==dict:
        new_d = {}
        for k,v in d.items():
            new_d[str(k)] = stringify_dict_keys(v)
    else:
        return d
    return new_d

class DessiaObject:
    _standalone_in_db = False
    def to_dict(self):
        if hasattr(self, 'Dict'):
            return self.Dict()
        elif hasattr(self, 'to_dict'):
            return self.to_dict()
        return self.to_dict
    
    @classmethod
    def dict_to_object(cls, dict_):
        if hasattr(cls, 'DictToObject'):
            return cls.DictToObject(dict_)
        raise NotImplementedError('Class has no dict_to_object/DictToObject method')