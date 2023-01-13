import unittest

from parameterized import parameterized

from dessia_common.utils.types import deserialize_typing, serialize_typing, MethodType, ClassMethodType, InstanceOf,\
    is_jsonable, is_sequence, is_list, is_tuple
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
        assert serialized_typing == 'typing.Type'
        deserialized_typing = deserialize_typing(serialized_typing)
        assert not deserialized_typing == test_typing


class TestIsJsonable(unittest.TestCase):
    # Some special cases above still use asserts
    @parameterized.expand([
        ({3: 2, 'attr': 45.3, 'attr_str': 'a str',
          'subobj': {'another_attr': [2.1, 1.3, 5.4]}}, True),
        (b'Test', False),
        ({3: b'Test'}, False)
    ])
    def test_jsonable(self, obj, jsonable):
        self.assertEqual(is_jsonable(obj), jsonable)

if __name__ == '__main__':
    unittest.main(verbosity=2)
