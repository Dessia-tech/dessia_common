import unittest

from parameterized import parameterized

from dessia_common.utils.types import deserialize_typing, serialize_typing, MethodType, ClassMethodType, InstanceOf
from dessia_common.files import BinaryFile, StringFile
from typing import List, Tuple, Type, Dict
from dessia_common.forms import StandaloneObject


class TestTypingSerializationValid(unittest.TestCase):
    # Some special cases above still use asserts
    @parameterized.expand([
        (List[StandaloneObject], 'List[dessia_common.forms.StandaloneObject]'),
        (List[int], 'List[__builtins__.int]'),
        # Nested Sequence
        (List[List[StandaloneObject]], 'List[List[dessia_common.forms.StandaloneObject]]'),
        (List[List[List[StandaloneObject]]], 'List[List[List[dessia_common.forms.StandaloneObject]]]'),

        # Dictionnaries
        (Dict[int, StandaloneObject], 'Dict[__builtins__.int, dessia_common.forms.StandaloneObject]'),
        # Types
        (Type, 'typing.Type'),
        (InstanceOf[StandaloneObject], 'InstanceOf[dessia_common.forms.StandaloneObject]'),
        (MethodType, 'dessia_common.typings.MethodType'),
        (ClassMethodType, 'dessia_common.typings.ClassMethodType'),
        # Files
        (StringFile, 'dessia_common.files.StringFile'),
        (BinaryFile, 'dessia_common.files.BinaryFile')
    ])
    def test_serialization(self, obj, str_obj):
        serialized = serialize_typing(obj)
        self.assertEqual(serialized, str_obj)
        self.assertEqual(obj, deserialize_typing(serialized))


    def test_serialization_special_cases(self):
        # Sequences
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

        # Types
        test_typing = Type[StandaloneObject]
        serialized_typing = serialize_typing(test_typing)
        assert serialized_typing == 'Type'
        deserialized_typing = deserialize_typing(serialized_typing)
        assert not deserialized_typing == test_typing



# === Typing Serialization ===

# Sequences

# test_typing = collections.abc.Iterator[StandaloneObject]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Iterator[dessia_common.forms.StandaloneObject]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

# Nested sequences

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

# test_typing = Dict[str, Dict[int, StandaloneObject]]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Dict[__builtins__.str, Dict[__builtins__.int, dessia_common.forms.StandaloneObject]]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

# Types

# test_typing = Subclass[StandaloneObject]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'InstanceOf[dessia_common.forms.StandaloneObject]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing

# test_typing = Union[StandaloneObject, str, int]
# serialized_typing = serialize_typing(test_typing)
# assert serialized_typing == 'Union[dessia_common.forms.StandaloneObject, __builtins__.str, __builtins__.int]'
# deserialized_typing = deserialize_typing(serialized_typing)
# assert deserialized_typing == test_typing