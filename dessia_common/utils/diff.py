#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:24:53 2021

@author: steven
"""

import math
import dessia_common as dc
from dessia_common.utils.types import isinstance_base_types, is_sequence, full_classname


def diff(value1, value2, path='#'):
    """
    Main function to get the diff of two objects
    :return: a tuple of a list of diff between the objects, missing keys in other object and invalid types
    """
    diff_values = []
    missing_keys_in_other_object = []
    invalid_types = []

    if is_sequence(value1) and is_sequence(value2):
        return sequence_diff(value1, value2, path=path)

    if not isinstance(value1, type(value2)):
        invalid_types.append(path)
        return diff_values, missing_keys_in_other_object, invalid_types

    if isinstance_base_types(value1):
        if isinstance(value1, float) and \
                math.isclose(value1, value2, abs_tol=dc.FLOAT_TOLERANCE):
            return diff_values, missing_keys_in_other_object, invalid_types
        if value1 != value2:
            diff_values.append((path, value1, value2))
        return diff_values, missing_keys_in_other_object, invalid_types
    if isinstance(value1, dict):
        return dict_diff(value1, value2, path=path)
    # elif hasattr(value1, '_data_eq'):

    # Should be object
    if hasattr(value1, '_data_eq'):
        # DessiaObject
        if value1._data_eq(value2):
            return [], [], []

        # Use same code snippet as in data_eq
        eq_dict = value1._serializable_dict()
        if 'name' in eq_dict:
            del eq_dict['name']

        other_eq_dict = value2._serializable_dict()
        return dict_diff(eq_dict, other_eq_dict, path=path)

    if value1 == value2:
        return [], [], []

    raise NotImplementedError('Undefined type in diff: {}'.format(type(value1)))


def dict_diff(dict1, dict2, path='#'):
    missing_keys_in_other_object = []
    diff_values = []
    invalid_types = []

    for key, value in dict1.items():
        path_key = '{}/{}'.format(path, key)
        if key not in dict2:
            missing_keys_in_other_object.append(key)
        else:
            diff_key, mkk, itk = diff(value, dict2[key], path=path_key)
            diff_values.extend(diff_key)
            missing_keys_in_other_object.extend(mkk)
            invalid_types.extend(itk)

    return diff_values, missing_keys_in_other_object, invalid_types


def sequence_diff(seq1, seq2, path='#'):
    diff_values = []
    missing_keys_in_other_object = []
    invalid_types = []

    if len(seq1) != len(seq2):
        diff_values.append((path, seq1, seq2))
    else:
        for i, (v1, v2) in enumerate(zip(seq1, seq2)):
            path_value = '{}/{}'.format(path, i)
            dv, mkv, itv = diff(v1, v2, path=path_value)
            # print('dvs', dv, v1, v2)
            diff_values.extend(dv)
            missing_keys_in_other_object.extend(mkv)
            invalid_types.extend(itv)
    return diff_values, missing_keys_in_other_object, invalid_types


def data_eq(value1, value2):
    if is_sequence(value1) and is_sequence(value2):
        return sequence_data_eq(value1, value2)

    if not isinstance(value2, type(value1))\
            and not isinstance(value1, type(value2)):
        return False

    if isinstance_base_types(value1):
        if isinstance(value1, float):
            return math.isclose(value1, value2, abs_tol=dc.FLOAT_TOLERANCE)

        return value1 == value2

    if isinstance(value1, dict):
        return dict_data_eq(value1, value2)

    # Else: its an object

    if full_classname(value1) != full_classname(value2):
        # print('full classname !=')
        return False

    # Test if _data_eq is customized
    if hasattr(value1, '_data_eq'):
        custom_method = (value1._data_eq.__code__
                         is not dc.DessiaObject._data_eq.__code__)
        if custom_method:
            return value1._data_eq(value2)

    # Not custom, use generic implementation
    eq_dict = value1._serializable_dict()
    if 'name' in eq_dict:
        del eq_dict['name']

    other_eq_dict = value2._serializable_dict()

    return dict_data_eq(eq_dict, other_eq_dict)


def dict_data_eq(dict1, dict2):

    for key, value in dict1.items():
        if key not in dict2:
            return False
        if not data_eq(value, dict2[key]):
            return False
    return True


def sequence_data_eq(seq1, seq2):
    if len(seq1) != len(seq2):
        return False

    for v1, v2 in zip(seq1, seq2):
        if not data_eq(v1, v2):
            # print('seq false')
            return False

    return True


def choose_hash(object_):
    if is_sequence(object_):
        return list_hash(object_)
    if isinstance(object_, dict):
        return dict_hash(object_)
    if isinstance(object_, str):
        return sum(ord(e) for e in object_)
    return hash(object_)


def list_hash(list_):
    return sum(choose_hash(e) for e in list_)


def dict_hash(dict_):
    hash_ = 0
    for key, value in dict_.items():
        hash_ += hash(key) + choose_hash(value)
    return hash_
