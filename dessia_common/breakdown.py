#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Code for breakdowns. """

import sys
from ast import literal_eval
import collections
import collections.abc
import numpy as npy

import dessia_common.serialization as dcs


def attrmethod_getter(object_, attr_methods):
    """
    Get method attributes from strings.

    Float with . in attributes are not handled.
    """
    # TODO: escape . inside ()
    for segment in attr_methods.split('.'):
        if '(' in segment:
            method, _, attributes = segment.partition('(')
            attributes = attributes[:-1]
            if attributes:
                object_ = getattr(object_, method)(literal_eval(attributes))
            else:
                object_ = getattr(object_, method)()
        else:
            object_ = getattr(object_, segment)
    return object_


def merge_breakdown_dicts(dict1, dict2):
    """ Merge strategy of breakdown dictionaries. """
    dict3 = dict1.copy()
    for class_name, refs in dict2.items():
        if class_name in dict3:
            # Decide by lower depth
            for obj, path in refs.items():
                if obj in dict3[class_name]:
                    if len(path.split('.')) < len(dict3[class_name][obj].split('.')):
                        dict3[class_name][obj] = path
                else:
                    dict3[class_name][obj] = path
        else:
            dict3[class_name] = refs
    return dict3


def breakdown(obj, path=''):
    """ Breakdown object as a dict. """
    bd_dict = {}

    if obj is None or isinstance(obj, (str, float, int, npy.ndarray)):
        return bd_dict

    if isinstance(obj, (list, tuple, set)):
        if path:
            path += '.'

        for i, element in enumerate(obj):
            path2 = f'{path}{i}'
            bd_dict = merge_breakdown_dicts(bd_dict, breakdown(element, path2))
    elif isinstance(obj, dict):
        if path:
            path += '.'

        for k, value in obj.items():
            path2 = path + str(k)
            bd_dict = merge_breakdown_dicts(bd_dict, breakdown(value, path2))
    else:
        # Put object and break it down
        if path and hasattr(obj, '__dict__'):  # avoid to get root object
            classname = obj.__class__.__name__
            if classname in bd_dict:
                if obj in bd_dict[classname]:
                    if len(path.split('.')) < len(bd_dict[classname][obj].split('.')):
                        bd_dict[classname][obj] = path
                else:
                    bd_dict[classname][obj] = path
            else:
                bd_dict[classname] = collections.OrderedDict()
                bd_dict[classname][obj] = path

        bd_dict = merge_breakdown_dicts(bd_dict, object_breakdown(obj, path=path))

    return bd_dict


def object_breakdown(obj, path=''):
    """ Return breakdown dict of object (no preliminary checks). """
    if path:
        path += '.'

    bd_dict = {}
    if isinstance(obj, dcs.SerializableObject):
        obj_dict = obj._serializable_dict()
    else:
        if hasattr(obj, '__dict__'):
            obj_dict = obj.__dict__
        else:
            obj_dict = {}

    for k, value in obj_dict.items():
        # dict after lists
        if not isinstance(value, (dict, list, tuple)):  # Should be object or builtins
            dict2 = breakdown(value, path=path + k)
            bd_dict = merge_breakdown_dicts(bd_dict, dict2)

    for k, value in obj_dict.items():
        # First lists and tuples
        if isinstance(value, (list, tuple)):
            dict2 = breakdown(value, path=path + k)
            bd_dict = merge_breakdown_dicts(bd_dict, dict2)

    for k, value in obj_dict.items():
        # dict after lists
        if isinstance(value, dict):
            dict2 = breakdown(value, path=path + k)
            bd_dict = merge_breakdown_dicts(bd_dict, dict2)
    return bd_dict


def deep_getsizeof(obj, ids=None):
    """
    Find the memory footprint of a Python object.

    This is a recursive function that drills down a Python object graph like a dictionary holding nested
    dictionaries with lists of lists and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each object inside a container as
    pointer only regardless of how big it really is.

    :param obj: the object
    :param ids:
    :return:
    """
    if ids is None:
        ids = set()

    dgso = deep_getsizeof
    if id(obj) in ids:
        return 0

    result = sys.getsizeof(obj)
    ids.add(id(obj))

    if isinstance(obj, str):
        return result

    if isinstance(obj, collections.abc.Mapping):
        return result + sum(dgso(k, ids) + dgso(v, ids) for k, v in obj.items())

    if isinstance(obj, collections.abc.Container):
        return result + sum(dgso(x, ids) for x in obj.__dict__.values())

    return result


def breakdown_analysis(obj):
    """ Return an analysis of the structure of the object. """
    obj_breakdown = object_breakdown(obj)

    stats = {"total_size": deep_getsizeof(obj)}

    number_objs = {}
    size_objs = {}
    meansize_objs = {}
    for class_name, class_objs in obj_breakdown.items():
        nobjs = len(class_objs)
        number_objs[class_name] = nobjs
        size_objs[class_name] = deep_getsizeof(class_objs)
        meansize_objs[class_name] = size_objs[class_name] / nobjs
    stats.update({"subobjects_number_by_class": number_objs,
                  "subobjects_size_by_class": size_objs,
                  "subobjects_meansize_by_class": meansize_objs})
    return stats
