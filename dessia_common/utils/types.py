#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Types tools. """

import warnings
from collections.abc import Iterator, Sequence
from typing import Any, Dict, List, Tuple, Type, Union, get_origin, get_args
import orjson
from dessia_common.abstract import CoreDessiaObject
from dessia_common.typings import InstanceOf, MethodType, ClassMethodType
from dessia_common.files import BinaryFile, StringFile
from dessia_common.schemas.core import TYPING_EQUIVALENCES, union_is_default_value, is_typing, serialize_annotation
from dessia_common.utils.helpers import get_python_class_from_class_name

SIMPLE_TYPES = [int, str]

TYPES_STRINGS = {int: 'int', float: 'float', bool: 'boolean', str: 'str', list: 'list', tuple: 'tuple', dict: 'dict'}

SEQUENCE_TYPINGS = ['List', 'Sequence', 'Iterable']

SERIALIZED_BUILTINS = ['float', 'builtins.float', 'int', 'builtins.int', 'str', 'builtins.str', 'bool', 'builtins.bool']


def is_classname_transform(string: str):
    """ Check if string is class name and return class if yes. """
    if '.' in string:
        split_string = string.split('.')
        if len(split_string) >= 2:
            try:
                class_ = get_python_class_from_class_name(string)
                return class_
            except (AttributeError, TypeError, ModuleNotFoundError, SyntaxError):
                return False
    return False


def is_jsonable(obj):
    """ Returns if the object can be dumped as it is in a JSON. """
    # First trying with orjson which is more efficient
    try:
        orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS).decode('utf-8')
        return True
    except Exception:
        return False


def is_serializable(_):
    """ Return True if object is deeply serializable as Dessia's standards, else False. """
    msg = "Function is_serializable has been moved to module serialization.py. Please use this one instead."
    raise NotImplementedError(msg)


def is_sequence(obj) -> bool:
    """
    Return True if object is sequence (but not string), else False.

    :param obj: Object to check
    :return: bool. True if object is a sequence but not a string. False otherwise
    """
    if not hasattr(obj, "__len__") or not hasattr(obj, "__getitem__"):
        # Performance improvements for trivial checks
        return False

    if is_list(obj) or is_tuple(obj):
        # Performance improvements for trivial checks
        return True
    return isinstance(obj, Sequence) and not isinstance(obj, str)


def is_list(obj) -> bool:
    """ Check if given obj is exactly of type list (not instance of). Used mainly for performance. """
    return obj.__class__ == list


def is_tuple(obj) -> bool:
    """ Check if given obj is exactly of type tuple (not instance of). Used mainly for performance. """
    return obj.__class__ == tuple


def is_builtin(type_):
    """ Return True if type_ is a simple python builtin (int, float, bool or str). """
    return type_ in TYPING_EQUIVALENCES


def is_simple(obj):
    """ Return True if given object is a int or a str or None. Used mainly for performance. """
    return obj is None or obj.__class__ in SIMPLE_TYPES


def isinstance_base_types(obj):
    """ Return True if the object is either a str, a float an int or None. """
    if is_simple(obj):
        # Performance improvements for trivial types
        return True
    return isinstance(obj, (str, float, int))


def is_dessia_file(obj):
    """ Wether object inherits from dessia files. """
    return isinstance(obj, (BinaryFile, StringFile))


def is_file_or_file_sequence(object_):
    """ Whether object inherits from dessia file or is a container of such objects. """
    is_file = is_dessia_file(object_)
    if not is_file and is_sequence(object_):
        for obj in object_:
            if is_file_or_file_sequence(obj):
                return True
    return is_file


def unfold_deep_annotation(typing_=None):
    """ Get origin (tuple, list,...) and arguments (type,...) from typing. """
    if is_typing(typing_):
        origin = get_origin(typing_)
        args = get_args(typing_)
        return origin, args
    return None, None


def serialize_typing(typing_):
    """ Compute a string from a type. """
    warnings.warn("Function serialize_typing is deprecated. Use serialize_annotation instead", DeprecationWarning)
    return serialize_annotation(typing_)


def type_from_argname(argname):
    """ Try and compute a type from an argument's name. """
    splitted_argname = argname.rsplit('.', 1)
    if argname:
        if splitted_argname[0] == '__builtins__':
            argname = f"builtins.{splitted_argname[1]}"
        return get_python_class_from_class_name(argname)
    return Any


TYPING_FROM_SERIALIZED_NAME = {"List": List, "Tuple": Tuple, "Iterator": Iterator, "Dict": Dict}


def deserialize_typing(serialized_typing: str):
    """ Compute a typing from a string. """
    # TODO : handling recursive deserialization
    if isinstance(serialized_typing, str):
        # TODO other builtins should be implemented
        if serialized_typing in SERIALIZED_BUILTINS:
            return deserialize_builtin_typing(serialized_typing)

        if serialized_typing in ["dessia_common.files.StringFile", "dessia_common.files.BinaryFile"]:
            return deserialize_file_typing(serialized_typing)

        if serialized_typing in ["dessia_common.typings.MethodType", "dessia_common.typings.ClassMethodType"]:
            return deserialize_method_typing(serialized_typing)

        if serialized_typing == "Type":
            return Type

        if '[' in serialized_typing:
            toptype, remains = serialized_typing.split('[', 1)
            full_argname = remains.rsplit(']', 1)[0]
            if "[" in full_argname:
                return TYPING_FROM_SERIALIZED_NAME[toptype][deserialize_typing(full_argname)]
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
        if toptype == "InstanceOf":
            return InstanceOf[type_from_argname(full_argname)]
        return get_python_class_from_class_name(serialized_typing)
    raise NotImplementedError(f'{serialized_typing} of type {type(serialized_typing)}')


def deserialize_tuple_typing(full_argname):
    """ Compute a tuple typing from a string. """
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
    """ Compute a file typing from a string. """
    if serialized_typing == "dessia_common.files.StringFile":
        return StringFile
    if serialized_typing == "dessia_common.files.BinaryFile":
        return BinaryFile
    raise NotImplementedError(f"File typing {serialized_typing} deserialization is not implemented")


def deserialize_method_typing(serialized_typing):
    """ Compute a method typing from a string. """
    if serialized_typing == "dessia_common.typings.MethodType":
        return MethodType
    if serialized_typing == "dessia_common.typings.ClassMethodType":
        return ClassMethodType
    raise NotImplementedError(f"Method typing {serialized_typing} deserialization is not implemented")


def deserialize_builtin_typing(serialized_typing):
    """ Compute a builtin typing from a string. """
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
    """ Return BSON validity (bool) and a hint (str). """
    if isinstance(value, (int, float, str)):
        return True, ''

    if value is None:
        return True, ''

    if isinstance(value, dict):
        for k, subvalue in value.items():
            # Key check
            if isinstance(k, str):
                if '.' in k:
                    return False, f'key {k} of dict is a string containing a ., which is forbidden'
            elif isinstance(k, float):
                return False, f'key {k} of dict is a float, which is forbidden'
            elif isinstance(k, int):
                if not allow_nonstring_keys:
                    log = f'key {k} of dict is an unsuported type {type(k)},' \
                          f'use allow_nonstring_keys=True to allow'
                    return False, log
            else:
                return False, f'key {k} of dict is an unsuported type {type(k)}'

            # Value Check
            v_valid, hint = is_bson_valid(value=subvalue, allow_nonstring_keys=allow_nonstring_keys)
            if not v_valid:
                return False, hint

    elif is_sequence(value):
        for subvalue in value:
            valid, hint = is_bson_valid(value=subvalue, allow_nonstring_keys=allow_nonstring_keys)
            if not valid:
                return valid, hint
    else:
        return False, f'Unrecognized type: {type(value)}'
    return True, ''


# TODO recursive_type and recursive_type functions look weird


def recursive_type(obj):
    """ What is the difference with serialize typing (?). """
    if isinstance(obj, tuple(list(TYPING_EQUIVALENCES.keys()) + [dict])):
        type_ = TYPES_STRINGS[type(obj)]
    elif isinstance(obj, CoreDessiaObject):
        type_ = obj.__module__ + '.' + obj.__class__.__name__
    elif hasattr(obj, 'output_type'):
        type_ = obj.output_type
    elif isinstance(obj, (list, tuple)):
        type_ = []
        for element in obj:
            type_.append(recursive_type(element))
    elif obj is None:
        type_ = None
    else:
        raise NotImplementedError(obj)
    return type_


def typematch(type_: Type, match_against: Type) -> bool:
    """
    Return whether type_ matches against match_against.

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
    """ Match type when type_ is a complex typing (List, Union, Tuple,...). """
    # Complex typing for the first type_. Cases : List, Tuple, Union
    if not is_typing(match_against):
        # Type matching is unilateral and match against should be wider than type_
        return False

    # Inspecting and healing types
    type_, type_origin, type_args = heal_type(type_)
    match_against, match_against_origin, match_against_args = heal_type(match_against)

    if type_origin != match_against_origin:
        # Being strict for now. Is there any other case than default values where this would be wrong ?
        return False

    if type_origin is list:
        # Can only have one argument, should match
        return typematch(type_args[0], match_against_args[0])

    if type_origin is tuple:
        # Order matters, all arguments should match
        return all(typematch(a, b) for a, b in zip(type_args, match_against_args))

    if type_origin is dict:
        # key type AND value type should match
        return typematch(type_args[0], match_against_args[0]) and typematch(type_args[1], match_against_args[1])

    if type_origin is Union:
        # type arguments must be a subset of match_against arguments set
        type_argsset = set(type_args)
        match_against_argsset = set(match_against_args)
        return type_argsset.issubset(match_against_argsset)

    # Otherwise, it is not implemented
    raise NotImplementedError(f"Type {type_} is a complex typing and cannot be matched against others yet")


def heal_type(type_: Type):
    """
    Inspect type and returns its parameters.

    For now, only checks whether the type is an 'Optional' / Union[T, NoneType], which should be flattened and not
    considered.

    Returns the cleaned type, origin and arguments.
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
    """ Check for specific cases of type matches and return a Boolean. """
    if type_ is int and match_against is float:
        return True
    # Not refactoring this as a one-liner for now, as more cases should be added in the future.
    return False
