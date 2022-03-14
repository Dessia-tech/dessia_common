#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from typing import List, Dict, Tuple, Union, Any, TextIO, BinaryIO, get_origin, get_args

import dessia_common as dc
from dessia_common.typings import Subclass, InstanceOf, MethodType, ClassMethodType

import json
import collections
from importlib import import_module


TYPING_EQUIVALENCES = {int: 'number', float: 'number', bool: 'boolean', str: 'string'}

TYPES_STRINGS = {int: 'int', float: 'float', bool: 'boolean', str: 'str',
                 list: 'list', tuple: 'tuple', dict: 'dict'}

SEQUENCE_TYPINGS = ['List', 'Sequence', 'Iterable']

TYPES_FROM_STRING = {'unicode': str, 'str': str, 'float': float, 'int': int, 'bool': bool}

SERIALIZED_BUILTINS = ['float', 'builtins.float', 'int', 'builtins.int', 'str', 'builtins.str', 'bool', 'builtins.bool']


def full_classname(object_, compute_for: str = 'instance'):
    if compute_for == 'instance':
        return object_.__class__.__module__ + '.' + object_.__class__.__name__
    elif compute_for == 'class':
        return object_.__module__ + '.' + object_.__name__
    else:
        msg = 'Cannot compute {} full classname for object {}'
        raise NotImplementedError(msg.format(compute_for, object_))


def is_jsonable(x):
    """
    returns if object can be dumped as it is in a json
    """
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def is_sequence(obj):
    """
    :param obj: Object to check
    :return: bool. True if object is a sequence but not a string.
                   False otherwise
    """
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


def is_builtin(type_):
    return type_ in TYPING_EQUIVALENCES


def isinstance_base_types(obj):
    """
    returns True if the object is either a str, a float a int or None
    """
    return isinstance(obj, str) or isinstance(obj, float) or isinstance(obj, int) or (obj is None)


def get_python_class_from_class_name(full_class_name):
    module_name, class_name = full_class_name.rsplit('.', 1)
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def unfold_deep_annotation(typing_=None):
    if is_typing(typing_):
        origin = get_origin(typing_)
        args = get_args(typing_)
        return origin, args
    return None, None


def is_typing(object_: Any):
    has_module = hasattr(object_, '__module__')
    if has_module:
        in_typings = object_.__module__ in ['typing', 'dessia_common.typings']
    else:
        return False
    has_origin = hasattr(object_, '__origin__')
    has_args = hasattr(object_, '__args__')
    return has_module and has_origin and has_args and in_typings


def serialize_typing(typing_):
    if is_typing(typing_):
        return serialize_typing_types(typing_)
    if isinstance(typing_, type):
        return full_classname(typing_, compute_for='class')
    if typing_ is TextIO:
        return "TextFile"
    if typing_ is BinaryIO:
        return "BinaryFile"
    return str(typing_)


def serialize_typing_types(typing_):
    origin = get_origin(typing_)
    args = get_args(typing_)
    if origin is Union:
        return serialize_union_typing(args)
    if origin is list:
        return f"List[{type_fullname(args[0])}]"
    if origin is tuple:
        argnames = ', '.join([type_fullname(a) for a in args])
        return f'Tuple[{argnames}]'
    if origin is collections.Iterator:
        return f"Iterator[{type_fullname(args[0])}]"
    if origin is dict:
        key_type = type_fullname(args[0])
        value_type = type_fullname(args[1])
        return f'Dict[{key_type}, {value_type}]'
    if origin is InstanceOf:
        return f'InstanceOf[{type_fullname(args[0])}]'
    if origin is Subclass:
        return f'Subclass[{type_fullname(args[0])}]'
    if origin is MethodType:
        return f'MethodType[{type_fullname(args[0])}]'
    if origin is ClassMethodType:
        return f'ClassMethodType[{type_fullname(args[0])}]'
    raise NotImplementedError(f"Serialization of typing {typing_} is not implemented")


def serialize_union_typing(args):
    if len(args) == 2 and type(None) in args:
        # This is a false Union => Is a default value set to None
        return serialize_typing(args[0])
    else:
        # Types union
        argnames = ', '.join([type_fullname(a) for a in args])
        return f'Union[{argnames}]'


def type_fullname(arg):
    if arg.__module__ == 'builtins':
        full_argname = '__builtins__.' + arg.__name__
    else:
        full_argname = serialize_typing(arg)
    return full_argname


def type_from_argname(argname):
    splitted_argname = argname.rsplit('.', 1)
    if argname:
        if splitted_argname[0] != '__builtins__':
            return get_python_class_from_class_name(argname)
        # TODO Check for dangerous eval
        return eval(splitted_argname[1])
    return Any


def deserialize_typing(serialized_typing):
    # TODO : handling recursive deserialization
    if isinstance(serialized_typing, str):
        # TODO other builtins should be implemented
        if serialized_typing in SERIALIZED_BUILTINS:
            return deserialize_builtin_typing(serialized_typing)

        if serialized_typing in ["TextFile", "BinaryFile"]:
            return deserialize_file_typing(serialized_typing)

        if '[' in serialized_typing:
            toptype, remains = serialized_typing.split('[', 1)
            full_argname = remains.rsplit(']', 1)[0]
        else:
            toptype = serialized_typing
            full_argname = ''
        if toptype == 'List':
            return List[type_from_argname(full_argname)]
        elif toptype == 'Tuple':
            return deserialize_tuple_typing(full_argname)
        elif toptype == 'Dict':
            args = full_argname.split(', ')
            key_type = type_from_argname(args[0])
            value_type = type_from_argname(args[1])
            return Dict[key_type, value_type]
        return get_python_class_from_class_name(serialized_typing)
    raise NotImplementedError('{} of type {}'.format(serialized_typing, type(serialized_typing)))


def deserialize_tuple_typing(full_argname):
    if ', ' in full_argname:
        args = full_argname.split(', ')
        if len(args) == 0:
            return Tuple
        elif len(args) == 1:
            type_ = type_from_argname(args[0])
            return Tuple[type_]
        elif len(set(args)) == 1:
            type_ = type_from_argname(args[0])
            return Tuple[type_, ...]
        else:
            raise TypeError("Heterogenous tuples are forbidden as types for workflow non-block variables.")
    return Tuple[type_from_argname(full_argname)]


def deserialize_file_typing(serialized_typing):
    if serialized_typing == "TextFile":
        return TextIO
    if serialized_typing == "BinaryFile":
        return BinaryIO
    raise NotImplementedError(f"File typing {serialized_typing} deserialization is not implemented")


def deserialize_builtin_typing(serialized_typing):
    if serialized_typing in ['float', 'builtins.float']:
        return float
    if serialized_typing in ['int', 'builtins.int']:
        return int
    if serialized_typing in ['str', 'builtins.str']:
        return str
    if serialized_typing in ['bool', 'builtins.bool']:
        return bool
    raise NotImplementedError(f"Builtin typing of {serialized_typing} deserialization is not implemented")


def is_bson_valid(value, allow_nonstring_keys=False) -> Tuple[bool, str]:
    """
    returns validity (bool) and a hint (str)
    """
    if isinstance(value, (int, float, str)):
        return True, ''

    if value is None:
        return True, ''

    if isinstance(value, dict):
        for k, v in value.items():
            # Key check
            if isinstance(k, str):
                if '.' in k:
                    log = 'key {} of dict is a string containing a .,' \
                          ' which is forbidden'
                    return False, log.format(k)
            elif isinstance(k, float):
                log = 'key {} of dict is a float, which is forbidden'
                return False, log.format(k)
            elif isinstance(k, int):
                if not allow_nonstring_keys:
                    log = 'key {} of dict is an unsuported type {},' \
                          ' use allow_nonstring_keys=True to allow'
                    return False, log.format(k, type(k))
            else:
                log = 'key {} of dict is an unsuported type {}'
                return False, log.format(k, type(k))

            # Value Check
            v_valid, hint = is_bson_valid(
                value=v, allow_nonstring_keys=allow_nonstring_keys
            )
            if not v_valid:
                return False, hint

    elif is_sequence(value):
        for v in value:
            valid, hint = is_bson_valid(
                value=v, allow_nonstring_keys=allow_nonstring_keys
            )
            if not valid:
                return valid, hint
    else:
        return False, f'Unrecognized type: {type(value)}'
    return True, ''

# TODO recursive_type and recursive_type functions look weird


def recursive_type(obj):
    if isinstance(obj, tuple(list(TYPING_EQUIVALENCES.keys()) + [dict])):
        type_ = TYPES_STRINGS[type(obj)]
    elif isinstance(obj, dc.DessiaObject):
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
