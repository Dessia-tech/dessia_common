#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module to compute differences between objects. """

import math
from typing import List

import numpy as npy
from dessia_common import FLOAT_TOLERANCE
import dessia_common.core as dc
from dessia_common.utils.types import isinstance_base_types, is_sequence, full_classname
from dessia_common.files import BinaryFile, StringFile


class DifferentValues:
    """ Contains info on diff different values. """

    def __init__(self, path, value1, value2):
        self.path = path
        self.value1 = value1
        self.value2 = value2

    def __repr__(self):
        return f'{self.path}: values differ {self.value1} / {self.value2}'


class MissingAttribute:
    """ Contains info on diff missing attribute. """

    def __init__(self, path: str, missing_in_first_object: bool):
        self.path = path
        self.missing_in_first_object = missing_in_first_object

    def __repr__(self):
        if self.missing_in_first_object:
            return f'{self.path}: missing in first object'

        return f'{self.path}: missing in second object'


class DifferentType:
    """ Contains info on diff different types. """

    def __init__(self, path: str, value1, value2):
        self.path = path
        self.value1 = value1
        self.value2 = value2

    def __repr__(self):
        return f'{self.path}: types differ {self.value1} / {self.value2}'


class Diff:
    """ Contains info on a performed diff analysis between two values. """

    def __init__(self, different_values: List[DifferentValues], missing_attributes: List[MissingAttribute],
                 invalid_types: List[DifferentType]):
        self.different_values = different_values
        self.missing_attributes = missing_attributes
        self.invalid_types = invalid_types

    def is_empty(self):
        """ Return True if diff is empty, else False. """
        return (not self.different_values) and (not self.missing_attributes) and (not self.invalid_types)

    def __repr__(self):
        if self.is_empty():
            return 'Objects are equal, diff is empty'

        object_print = ''
        if self.different_values:
            object_print += 'Different values:\t'
            for diff_value in self.different_values:
                object_print += f'\t{diff_value}\n'

        if self.missing_attributes:
            object_print += 'Missing attributes:\t'
            for diff_value in self.missing_attributes:
                object_print += f'\t{diff_value}\n'

        if self.invalid_types:
            object_print += 'Invalid types:\n'
            for diff_value in self.invalid_types:
                object_print += f'\t{diff_value}\n'
        return object_print

    def __add__(self, other_diff):
        return Diff(self.different_values + other_diff.different_values,
                    self.missing_attributes + other_diff.missing_attributes,
                    self.invalid_types + other_diff.invalid_types)


def diff(value1, value2, path='#'):
    """
    Main function to get the diff of two objects.

    :return: a tuple of a list of diff between the objects, missing keys in other object and invalid types
    """
    diff_values = []
    missing_keys_in_other_object = []
    invalid_types = []

    if is_sequence(value1) and is_sequence(value2):
        return sequence_diff(value1, value2, path=path)

    if not isinstance(value1, type(value2)):
        # print('tv2', type(value2), type(value1))
        # print(isinstance(value1, type(value2)))
        # invalid_types.append(path)
        invalid_types.append(DifferentType(path, value1, value2))
        return Diff(diff_values, missing_keys_in_other_object, invalid_types)

    if isinstance_base_types(value1):
        if isinstance(value1, float) and math.isclose(value1, value2, abs_tol=FLOAT_TOLERANCE):
            return Diff(diff_values, missing_keys_in_other_object, invalid_types)
        if value1 != value2:
            # diff_values.append((path, value1, value2))
            diff_values.append(DifferentValues(path, value1, value2))
        return Diff(diff_values, missing_keys_in_other_object, invalid_types)
    if isinstance(value1, dict):
        return dict_diff(value1, value2, path=path)

    # Should be object
    if hasattr(value1, '_data_eq'):
        # DessiaObject
        if value1._data_eq(value2):
            return Diff([], [], [])

        # Use same code snippet as in data_eq
        eq_dict = value1._serializable_dict()
        if 'name' in eq_dict:
            del eq_dict['name']

        other_eq_dict = value2._serializable_dict()
        return dict_diff(eq_dict, other_eq_dict, path=path)

    if value1 == value2:
        return Diff([], [], [])

    raise NotImplementedError(f'Undefined type in diff: {type(value1)}')


def dict_diff(dict1, dict2, path='#'):
    """ Returns diff between two dicts. """
    diff_object = Diff([], [], [])

    dict_ = dict1
    other_dict = dict2
    first_object = False
    if len(dict2) > len(dict1):
        dict_ = dict2
        other_dict = dict1
        first_object = True
    for key, value in dict_.items():
        path_key = f'{path}/{key}'
        if key not in other_dict:
            diff_object.missing_attributes.append(MissingAttribute(path=path_key, missing_in_first_object=first_object))
        else:
            diff_key = diff(value, other_dict[key], path=path_key)
            diff_object += diff_key
    return diff_object


def sequence_diff(seq1, seq2, path='#'):
    """ Returns diff between two sequences. """
    seq_diff = Diff([], [], [])

    if len(seq1) != len(seq2):
        # diff_values.append((path, seq1, seq2))
        seq_diff.different_values.append(DifferentValues(path, seq1, seq2))
    else:
        for i, (v1, v2) in enumerate(zip(seq1, seq2)):
            path_value = f'{path}/{i}'
            diff_value = diff(v1, v2, path=path_value)
            seq_diff += diff_value
    return seq_diff


def data_eq(value1, value2):
    """ Returns if two values are equal on data equality. """
    if is_sequence(value1) and is_sequence(value2):
        return sequence_data_eq(value1, value2)

    if isinstance(value1, npy.int64) or isinstance(value2, npy.int64):
        return value1 == value2

    if isinstance(value1, npy.float64) or isinstance(value2, npy.float64):
        return math.isclose(value1, value2, abs_tol=FLOAT_TOLERANCE)

    if not isinstance(value2, type(value1))\
            and not isinstance(value1, type(value2)):
        return False

    if isinstance_base_types(value1):
        if isinstance(value1, float):
            return math.isclose(value1, value2, abs_tol=FLOAT_TOLERANCE)

        return value1 == value2

    if isinstance(value1, dict):
        return dict_data_eq(value1, value2)

    if isinstance(value1, (BinaryFile, StringFile)):
        return value1 == value2

    if isinstance(value1, type):
        return full_classname(value1) == full_classname(value2)

    # Else: its an object
    if full_classname(value1) != full_classname(value2):
        # print('full classname !=')
        return False

    # Test if _data_eq is customized
    if hasattr(value1, '_data_eq'):
        custom_method = value1._data_eq.__code__ is not dc.DessiaObject._data_eq.__code__
        if custom_method:
            return value1._data_eq(value2)

    # Not custom, use generic implementation
    eq_dict = value1._data_eq_dict()
    if 'name' in eq_dict:
        del eq_dict['name']

    other_eq_dict = value2._data_eq_dict()

    return dict_data_eq(eq_dict, other_eq_dict)


def dict_data_eq(dict1, dict2):
    """ Returns if two dicts are equal on data equality. """
    for key, value in dict1.items():
        if key not in dict2:
            return False
        if not data_eq(value, dict2[key]):
            return False
    return True


def sequence_data_eq(seq1, seq2):
    """ Returns if two sequences are equal on data equality. """
    if len(seq1) != len(seq2):
        return False

    for v1, v2 in zip(seq1, seq2):
        if not data_eq(v1, v2):
            # print('seq false')
            return False

    return True


def choose_hash(object_):
    """ Base function to return hash. """
    if is_sequence(object_):
        return sequence_hash(object_)
    if isinstance(object_, dict):
        return dict_hash(object_)
    if isinstance(object_, str):
        return sum(ord(e) for e in object_)
    return hash(object_)


def sequence_hash(sequence):
    """
    Return hash of a sequence value.

    Only checks for first and last elements hashes if defined for performance purpose.
    It also looks that previous sequence hash method was lest efficient as the sum of all hashes in sequence
    returned less unique values for normally different sequences.
    """
    if not sequence:
        return 0

    # Recursively compute hash of first and last element for performance purpose
    hash_ = len(sequence)*choose_hash(sequence[0])
    if len(sequence) > 1:
        hash_ += 5381*choose_hash(sequence[-1])
    return hash_


def dict_hash(dict_):
    """
    Returns hash of a dict value.

    If keys are orderable, only checks for first and last elements hashes if defined for performance purpose.
    """
    if not dict_:
        return 0

    hash_ = 0
    try:
        # Try and sort keys in order to get first and last elements.
        sorted_keys = sorted(dict_.keys())
    except TypeError:
        # Old less performant hash for non orderable keys.
        for key, value in dict_.items():
            hash_ += hash(key) + choose_hash(value)
        return hash_

    # Recursively compute hash of first and last element for performance purpose
    first_key = sorted_keys[0]
    hash_ += len(dict_) * choose_hash(dict_[first_key])
    if len(dict_) > 1:
        last_key = sorted_keys[-1]
        hash_ += 313 * choose_hash(dict_[last_key])
    return hash_
