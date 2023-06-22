#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Serialization Tools. """


import warnings
import dessia_common.serialization as dcs


def serialize(value):
    """ Main function for serialization without pointers. Calls recursively itself. """
    warnings.warn("Module serialization.py have been moved outside utils. Please use it instead")
    return dcs.serialize(value)


def serialize_with_pointers(value, memo=None, path='#', id_method=True, id_memo=None):
    """ Main function for serialization with pointers. """
    warnings.warn("Module serialization.py have been moved outside utils. Please use it instead")
    return dcs.serialize_with_pointers(value=value, memo=memo, path=path, id_method=id_method, id_memo=id_memo)


def deserialize(serialized_element, sequence_annotation: str = 'List', global_dict=None,
                pointers_memo=None, path: str = '#'):
    """ Main function for deserialization, handle pointers. """
    warnings.warn("Module serialization.py have been moved outside utils. Please use it instead")
    return dcs.deserialize(serialized_element=serialized_element, sequence_annotation=sequence_annotation,
                           global_dict=global_dict, pointers_memo=pointers_memo, path=path)


def dict_to_object(dict_, class_=None, force_generic: bool = False, global_dict=None, pointers_memo=None, path='#'):
    """ Transform a dictionary to an object. """
    warnings.warn("Module serialization.py have been moved outside utils. Please use it instead")
    return dcs.dict_to_object(dict_=dict_, class_=class_, force_generic=force_generic, global_dict=global_dict,
                              pointers_memo=pointers_memo, path=path)
