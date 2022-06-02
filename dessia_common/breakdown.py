#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
from ast import literal_eval
import collections
import numpy as npy

import dessia_common
from dessia_common.utils.types import is_sequence


def attrmethod_getter(object_, attr_methods):
    """
    Float with . in attributes are not handled
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


class ExtractionError(Exception):
    pass


def extract_from_object(object_, segment):
    if is_sequence(object_):
        try:
            return object_[int(segment)]
        except ValueError:
            message_error = (f'Cannot extract segment {segment} from object {{str(object_)[:500]}}:'
                             + 'segment is not a sequence index')
            raise ExtractionError(message_error)

    if isinstance(object_, dict):
        if segment in object_:
            return object_[segment]

        if segment.isdigit():
            intifyed_segment = int(segment)
            if intifyed_segment in object_:
                return object_[intifyed_segment]
            if segment in object_:
                return object_[segment]

            raise ExtractionError(f'Cannot extract segment {segment} from object {str(object_)[:200]}')

        # should be a tuple
        if segment.startswith('(') and segment.endswith(')') and ',' in segment:
            key = []
            for subsegment in segment.strip('()').replace(' ', '').split(','):
                if subsegment.isdigit():
                    subkey = int(subsegment)
                else:
                    subkey = subsegment
                key.append(subkey)
            return object_[tuple(key)]
        # else:
        message_error = f'Cannot extract segment {segment} from object {str(object_)[:500]}'
        raise ExtractionError(message_error)

    # Finally, it is a regular object
    return getattr(object_, segment)


def get_in_object_from_path(object_, path):
    segments = path.lstrip('#/').split('/')
    element = object_
    for segment in segments:
        if isinstance(element, dict) and '$ref' in element:
            # Going down in the object and it is a reference
            # Evaluating subreference
            try:
                element = get_in_object_from_path(object_, element['$ref'])
            except RecursionError as err:
                err_msg = f'Cannot get segment {segment} from path {path} in element {str(element)[:500]}'
                raise RecursionError(err_msg) from err
        try:
            element = extract_from_object(element, segment)
        except ExtractionError:
            err_msg = f'Cannot get segment {segment} from path {path} in element {str(element)[:500]}'
            raise ExtractionError(err_msg)

    return element


def merge_breakdown_dicts(dict1, dict2):
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
            # dict3[class_name].update(refs)
        else:
            dict3[class_name] = refs
    return dict3


def breakdown(obj, path=''):
    bd_dict = {}
    if obj is None:
        return bd_dict

    if (isinstance(obj, str) or isinstance(obj, float) or isinstance(obj, int)):
        return bd_dict

    if isinstance(obj, npy.ndarray):
        return bd_dict

    if isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
        if path:
            path += '.'

        for i, element in enumerate(obj):
            path2 = path + '{}'.format(i)
            bd_dict = merge_breakdown_dicts(bd_dict, breakdown(element, path2))
    elif isinstance(obj, dict):
        if path:
            path += '.'

        for k, value in obj.items():
            path2 = path + str(k)
            bd_dict = merge_breakdown_dicts(bd_dict, breakdown(value, path2))
    else:
        # Put object and break it down
        if path:  # avoid to get root object
            if hasattr(obj, '__dict__'):
                if obj.__class__.__name__ in bd_dict:
                    if obj in bd_dict[obj.__class__.__name__]:
                        if len(path.split('.')) < len(bd_dict[obj.__class__.__name__][obj].split('.')):
                            bd_dict[obj.__class__.__name__][obj] = path
                    else:
                        bd_dict[obj.__class__.__name__][obj] = path
                else:
                    bd_dict[obj.__class__.__name__] = collections.OrderedDict()
                    bd_dict[obj.__class__.__name__][obj] = path

        bd_dict = merge_breakdown_dicts(bd_dict, object_breakdown(obj, path=path))

    return bd_dict


def object_breakdown(obj, path=''):
    """
    Return breakdown dict of object (no preliminary checks)
    """
    if path:
        path += '.'

    bd_dict = {}
    if isinstance(obj, dessia_common.core.DessiaObject):
        obj_dict = obj._serializable_dict()
    else:
        if hasattr(obj, '__dict__'):
            obj_dict = obj.__dict__
        else:
            obj_dict = {}

    for k, value in obj_dict.items():
        # dict after lists
        if not (isinstance(value, dict)
                or isinstance(value, list)
                or isinstance(value, tuple)
                ):  # Should be object or builtins
            dict2 = breakdown(value, path=path + k)
            bd_dict = merge_breakdown_dicts(bd_dict, dict2)

    for k, value in obj_dict.items():
        # First lists and tuples
        if isinstance(value, list) or isinstance(value, tuple):
            dict2 = breakdown(value, path=path + k)
            bd_dict = merge_breakdown_dicts(bd_dict, dict2)

    for k, value in obj_dict.items():
        # dict after lists
        if isinstance(value, dict):
            dict2 = breakdown(value, path=path + k)
            bd_dict = merge_breakdown_dicts(bd_dict, dict2)
    return bd_dict


def deep_getsizeof(obj, ids=None):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
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

    # if isinstance(o, collections.Mapping):
    #     return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())

    if isinstance(obj, collections.Mapping):
        return result + sum(dgso(k, ids) + dgso(v, ids) for k, v in obj.items())

    if isinstance(obj, collections.Container):
        return result + sum(dgso(x, ids) for x in obj.__dict__.values())

    return result


def breakdown_analysis(obj):
    obj_breakdown = object_breakdown(obj)

    stats = {}
    stats['total_size'] = deep_getsizeof(obj)

    number_objs = {}
    size_objs = {}
    meansize_objs = {}
    for class_name, class_objs in obj_breakdown.items():
        nobjs = len(class_objs)
        number_objs[class_name] = nobjs
        size_objs[class_name] = deep_getsizeof(class_objs)
        meansize_objs[class_name] = size_objs[class_name] / nobjs
    stats['subobjects_number_by_class'] = number_objs
    stats['subobjects_size_by_class'] = size_objs
    stats['subobjects_meansize_by_class'] = meansize_objs
    return stats
