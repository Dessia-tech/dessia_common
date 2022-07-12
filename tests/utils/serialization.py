#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:43:53 2021

@author: steven
"""

from dessia_common.utils.diff import dict_diff
from dessia_common.utils.types import deserialize_typing, serialize_typing, MethodType, ClassMethodType,\
    Subclass, InstanceOf, Union
from dessia_common.files import BinaryFile, StringFile
from typing import List, Tuple, Type, Dict
from dessia_common.forms import StandaloneObject

import collections.abc


ds1 = {'key': 'val1'}
ds2 = {'key': 'val2'}

path, val1, val2 = dict_diff(ds1, ds2)[0][0]
assert path == '#/key'
assert val1 == 'val1'
assert val2 == 'val2'

# === Typing Serialization ===
# Sequences
test_typing = List[StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'List[dessia_common.forms.StandaloneObject]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

test_typing = List[int]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'List[__builtins__.int]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

test_typing = Tuple[int, StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'Tuple[__builtins__.int, dessia_common.forms.StandaloneObject]'
try:
    deserialized_typing = deserialize_typing(serialized_typing)
except TypeError:
    # Workflow cannot have heterogeneous tuples for non_block_variables ???
    pass

test_typing = Tuple[StandaloneObject, StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'Tuple[dessia_common.forms.StandaloneObject, dessia_common.forms.StandaloneObject]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == Tuple[StandaloneObject, ...]

test_typing = collections.abc.Iterator[StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'Iterator[dessia_common.forms.StandaloneObject]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

# Nested sequences
test_typing = List[List[StandaloneObject]]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'List[List[dessia_common.forms.StandaloneObject]]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

test_typing = List[List[List[StandaloneObject]]]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'List[List[List[dessia_common.forms.StandaloneObject]]]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

# test_typing = Tuple[Tuple[int, StandaloneObject]]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Tuple[Tuple[__builtins__.int, dessia_common.forms.StandaloneObject]]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing
#
# test_typing = Tuple[Tuple[StandaloneObject, StandaloneObject]]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Tuple[Tuple[dessia_common.forms.StandaloneObject, dessia_common.forms.StandaloneObject]]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

# Dictionnaries
test_typing = Dict[int, StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'Dict[__builtins__.int, dessia_common.forms.StandaloneObject]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

# test_typing = Dict[str, Dict[int, StandaloneObject]]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Dict[__builtins__.str, Dict[__builtins__.int, dessia_common.forms.StandaloneObject]]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

# Types
test_typing = Type
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'typing.Type'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

test_typing = Type[StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'Type'
deserialized_typing = deserialize_typing(serialized_typing)
assert not deserialized_typing == test_typing

test_typing = InstanceOf[StandaloneObject]
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'InstanceOf[dessia_common.forms.StandaloneObject]'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

# test_typing = Subclass[StandaloneObject]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'InstanceOf[dessia_common.forms.StandaloneObject]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

test_typing = MethodType
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'MethodType'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

test_typing = ClassMethodType
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'ClassMethodType'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

# test_typing = Union[StandaloneObject, str, int]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Union[dessia_common.forms.StandaloneObject, __builtins__.str, __builtins__.int]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

# Files
test_typing = StringFile
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'StringFile'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing

test_typing = BinaryFile
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'BinaryFile'
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing


print("TYPING SERIALISATION TESTS PASSED")
