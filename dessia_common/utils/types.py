#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from typing import List, Dict, Type, Tuple, Union, Any, TextIO, BinaryIO, \
    get_type_hints, get_origin, get_args


from dessia_common.typings import Measure, JsonSerializable,\
    Subclass, InstanceOf, MethodType, ClassMethodType
    
import json
import collections
from importlib import import_module


TYPING_EQUIVALENCES = {int: 'number', float: 'number',
                       bool: 'boolean', str: 'string'}


def full_classname(object_, compute_for: str = 'instance'):
    if compute_for == 'instance':
        return object_.__class__.__module__ + '.' + object_.__class__.__name__
    elif compute_for == 'class':
        return object_.__module__ + '.' + object_.__name__
    else:
        msg = 'Cannot compute {} full classname for object {}'
        raise NotImplementedError(msg.format(compute_for, object_))
        

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def is_sequence(obj):
    """
    :param obj: Object to check
    :return: bool. True if object is a sequence but not a string.
                   False otherwise
    """
    return isinstance(obj, collections.abc.Sequence)\
        and not isinstance(obj, str)


def is_builtin(type_):
    return type_ in TYPING_EQUIVALENCES


def isinstance_base_types(obj):
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
        origin = get_origin(typing_)
        args = get_args(typing_)
        if origin is Union:
            if len(args) == 2 and type(None) in args:
                # This is a false Union => Is a default value set to None
                return serialize_typing(args[0])
            else:
                # Types union
                argnames = ', '.join([type_fullname(a) for a in args])
                return 'Union[{}]'.format(argnames)
        elif origin is list:
            return 'List[' + type_fullname(args[0]) + ']'
        elif origin is tuple:
            argnames = ', '.join([type_fullname(a) for a in args])
            return 'Tuple[{}]'.format(argnames)
        elif origin is collections.Iterator:
            return 'Iterator[' + type_fullname(args[0]) + ']'
        elif origin is dict:
            key_type = type_fullname(args[0])
            value_type = type_fullname(args[1])
            return 'Dict[{}, {}]'.format(key_type, value_type)
        elif origin is InstanceOf:
            return 'InstanceOf[{}]'.format(type_fullname(args[0]))
        elif origin is Subclass:
            return 'Subclass[{}]'.format(type_fullname(args[0]))
        elif origin is MethodType:
            return 'MethodType[{}]'.format(type_fullname(args[0]))
        elif origin is ClassMethodType:
            return 'ClassMethodType[{}]'.format(type_fullname(args[0]))
        else:
            msg = 'Serialization of typing {} is not implemented'
            raise NotImplementedError(msg.format(typing_))
    if isinstance(typing_, type):
        return full_classname(typing_, compute_for='class')
    if typing_ is TextIO:
        return "TextFile"
    if typing_ is BinaryIO:
        return "BinaryFile"
    return str(typing_)


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
        if serialized_typing in ['float', 'builtins.float']:
            return float
        if serialized_typing in ['int', 'builtins.int']:
            return int
        if serialized_typing in ['str', 'builtins.str']:
            return str
        if serialized_typing in ['bool', 'builtins.bool']:
            return bool

        if serialized_typing == "TextFile":
            return TextIO
        if serialized_typing == "BinaryFile":
            return BinaryIO

        if '[' in serialized_typing:
            toptype, remains = serialized_typing.split('[', 1)
            full_argname = remains.rsplit(']', 1)[0]
        else:
            toptype = serialized_typing
            full_argname = ''
        if toptype == 'List':
            return List[type_from_argname(full_argname)]
        elif toptype == 'Tuple':
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
                    msg = ("Heterogenous tuples are forbidden as types for"
                           "workflow non-block variables.")
                    raise TypeError(msg)
            return Tuple[type_from_argname(full_argname)]
        elif toptype == 'Dict':
            args = full_argname.split(', ')
            key_type = type_from_argname(args[0])
            value_type = type_from_argname(args[1])
            return Dict[key_type, value_type]
        # elif splitted_type[0] == 'Union':
        #     args = full_argname.split(', ')
        return get_python_class_from_class_name(serialized_typing)
    raise NotImplementedError('{} of type {}'.format(serialized_typing,
                                                     type(serialized_typing)))