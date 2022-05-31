#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Type, Union, TextIO, BinaryIO, get_origin, get_args

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

_PYTHON_CLASS_CACHE = {}


def full_classname(object_, compute_for: str = 'instance'):
    if compute_for == 'instance':
        return object_.__class__.__module__ + '.' + object_.__class__.__name__
    if compute_for == 'class':
        return object_.__module__ + '.' + object_.__name__

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
    cached_value = _PYTHON_CLASS_CACHE.get(full_class_name, None)
    if cached_value is not None:
        return cached_value

    module_name, class_name = full_class_name.rsplit('.', 1)
    module = import_module(module_name)
    class_ = getattr(module, class_name)

    # Storing in cache
    _PYTHON_CLASS_CACHE[full_class_name] = class_
    return class_


def unfold_deep_annotation(typing_=None):
    if is_typing(typing_):
        origin = get_origin(typing_)
        args = get_args(typing_)
        return origin, args
    return None, None


def is_typing(object_: Any):
    has_module = hasattr(object_, '__module__')
    has_origin = hasattr(object_, '__origin__')
    has_args = hasattr(object_, '__args__')
    return has_module and has_origin and has_args


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
    if origin is type:
        return "Type"
    raise NotImplementedError(f"Serialization of typing {typing_} is not implemented")


def serialize_union_typing(args):
    if len(args) == 2 and type(None) in args:
        # This is a false Union => Is a default value set to None
        return serialize_typing(args[0])

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
        return literal_eval(splitted_argname[1])
    return Any


def deserialize_typing(serialized_typing):
    # TODO : handling recursive deserialization
    if isinstance(serialized_typing, str):
        # TODO other builtins should be implemented
        if serialized_typing in SERIALIZED_BUILTINS:
            return deserialize_builtin_typing(serialized_typing)

        if serialized_typing in ["TextFile", "BinaryFile"]:
            return deserialize_file_typing(serialized_typing)

        if serialized_typing == "Type":
            return Type

        if '[' in serialized_typing:
            toptype, remains = serialized_typing.split('[', 1)
            full_argname = remains.rsplit(']', 1)[0]
        else:
            toptype = serialized_typing
            full_argname = ''
        if toptype == 'List':
            return List[type_from_argname(full_argname)]
        if toptype == 'Tuple':
            return deserialize_tuple_typing(full_argname)
        if toptype == 'Dict':
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
        if len(args) == 1:
            type_ = type_from_argname(args[0])
            return Tuple[type_]
        if len(set(args)) == 1:
            type_ = type_from_argname(args[0])
            return Tuple[type_, ...]

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
        for k, subvalue in value.items():
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
                value=subvalue, allow_nonstring_keys=allow_nonstring_keys
            )
            if not v_valid:
                return False, hint

    elif is_sequence(value):
        for subvalue in value:
            valid, hint = is_bson_valid(
                value=subvalue, allow_nonstring_keys=allow_nonstring_keys
            )
            if not valid:
                return valid, hint
    else:
        return False, f'Unrecognized type: {type(value)}'
    return True, ''

# TODO recursive_type and recursive_type functions look weird


def recursive_type(obj):
    """
    What is the difference with serialize typing?
    """

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


def union_is_default_value(typing_: Type) -> bool:
    """
    Union typings can be False positives.
    An argument of a function that has a default_value set to None is Optional[T],
    which is an alias for Union[T, NoneType]. This function checks if this is the case.
    """
    args = get_args(typing_)
    return len(args) == 2 and type(None) in args


def typematch(type_: Type, match_against: Type) -> bool:
    """
        Return wether type_ matches against match_against.
        match_against needs to be "wider" than type_, and the check is not bilateral
    """
    # TODO Implement a more intelligent check for Unions : Union[T, U] should match against Union[T, U, V]
    # TODO Implement a check for Dict
    if type_ == match_against or match_against is Any or particular_typematches(type_, match_against):
        # Trivial cases. If types are strictly equal, then it should pass straight away
        return True

    if is_typing(type_):
        return complex_first_type_match(type_, match_against)

    if not is_typing(match_against):
        # type_ and match_against aren't complex : check for subclass only
        if issubclass(type_, match_against):
            return True

    # type_ is not complex and match_against is
    match_against, origin, args = heal_type(match_against)
    if origin is Union:
        matches = [typematch(type_, subtype) for subtype in args]
        return any(matches)
    return False


def complex_first_type_match(type_: Type, match_against: Type) -> bool:
    """
    Match type when type_ is a complex typing (List, Union, Tuple,...)
    """
    # Complex typing for the first type_. Cases : List, Tuple, Union
    if not is_typing(match_against):
        # Type matching is unilateral and match against should be more open than type_
        return False

    # Inspecting and healing types
    type_, type_origin, type_args = heal_type(type_)
    match_against, match_against_origin, match_against_args = heal_type(match_against)

    if type_origin != match_against_origin:
        # Being strict for now. Is there any other case than default values where this would be wrong ?
        return False

    if type_origin is list:
        # Can only have one arg, should match
        return typematch(type_args[0], match_against_args[0])

    if type_origin is tuple:
        # Order matters, all args should match
        return all(typematch(a, b) for a, b in zip(type_args, match_against_args))

    if type_origin is dict:
        # key type AND value type should match
        return typematch(type_args[0], match_against_args[0]) and typematch(type_args[1], match_against_args[1])

    if type_origin is Union:
        # type args must be a subset of match_against args set
        type_argsset = set(type_args)
        match_against_argsset = set(match_against_args)
        return type_argsset.issubset(match_against_argsset)

    # Otherwise, it is not implemented
    raise NotImplementedError(f"Type {type_} is a complex typing and cannot be matched against others yet")


def heal_type(type_: Type):
    """
    Inspect type and returns its params

    For now, only checks wether the type is an 'Optional' / Union[T, NoneType],
    which should be flattened and not considered

    returns the cleaned type, origin and args
    """
    type_origin = get_origin(type_)
    type_args = get_args(type_)
    if type_origin is Union:
        # Check for default values false positive
        if union_is_default_value(type_):
            type_ = type_args[0]
            type_origin = get_origin(type_)
            type_args = get_args(type_)
    return type_, type_origin, type_args


def particular_typematches(type_: Type, match_against: Type) -> bool:
    """
    Checks for specific cases of typematches and returns and boolean
    """
    if type_ is int and match_against is float:
        return True
    return False
