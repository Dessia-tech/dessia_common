from typing import Union, Type, List, Dict, Tuple
from dessia_common.schemas.core import deserialize_annotation, extract_args, serialize_annotation
from dessia_common.forms import StandaloneObject
from dessia_common.tests import Model
from dessia_common.measures import Distance
from dessia_common.typings import InstanceOf, MethodType, ClassMethodType, AttributeType, ClassAttributeType
from dessia_common.files import StringFile, BinaryFile
import unittest
from parameterized import parameterized


class TestAnnotationSerializationValid(unittest.TestCase):
    @parameterized.expand([
        ("int", ["int"]),
        ("dessia_common.forms.StandaloneObject", ["dessia_common.forms.StandaloneObject"]),
        ("List[dessia_common.forms.StandaloneObject]", ["List[dessia_common.forms.StandaloneObject]"]),
        ("List[List[dessia_common.forms.StandaloneObject]]", ["List[List[dessia_common.forms.StandaloneObject]]"]),
        ("int, dessia_common.forms.StandaloneObject", ["int", "dessia_common.forms.StandaloneObject"]),
        ("int, str", ["int", "str"]),
        ("int,Tuple[dessia_common.forms.StandaloneObject,str]",
         ["int", "Tuple[dessia_common.forms.StandaloneObject,str]"]),
        ("Tuple[dessia_common.forms.StandaloneObject,str], int",
         ["Tuple[dessia_common.forms.StandaloneObject,str]", "int"]),
        ("int,Tuple[dessia_common.forms.StandaloneObject, str]",
         ["int", "Tuple[dessia_common.forms.StandaloneObject,str]"]),
        ("int, Tuple[int, Tuple[Tuple[int, float], int]], int",
         ["int", "Tuple[int,Tuple[Tuple[int,float],int]]", "int"]),
        ("int, Tuple[int, Tuple[Tuple[int, float], int]], int,List[package.module.class]",
         ["int", "Tuple[int,Tuple[Tuple[int,float],int]]", "int", "List[package.module.class]"]),
        ("int, Tuple[int, Tuple[Tuple[int, float], int]], int,List[package.module.class], int",
         ["int", "Tuple[int,Tuple[Tuple[int,float],int]]", "int", "List[package.module.class]", "int"]),
    ])
    def test_args_extraction(self, rawstring, extracted_sequence):
        self.assertEqual(extract_args(rawstring), extracted_sequence)

    @parameterized.expand([
        # Simple types
        (int, "int"),
        (str, "str"),
        (float, "float"),
        (StandaloneObject, "dessia_common.forms.StandaloneObject"),
        (Distance, "dessia_common.measures.Distance"),

        # Tuple
        (Tuple[int], "Tuple[int]"),
        (Tuple[int, str], "Tuple[int, str]"),
        (Tuple[int, ...], "Tuple[int, ...]"),
        (Tuple[int, StandaloneObject], "Tuple[int, dessia_common.forms.StandaloneObject]"),
        (Tuple[int, Tuple[StandaloneObject, str]], "Tuple[int, Tuple[dessia_common.forms.StandaloneObject, str]]"),

        # Union
        (Union[StandaloneObject, Model], "Union[dessia_common.forms.StandaloneObject, dessia_common.tests.Model]"),

        # List
        (List[int], "List[int]"),
        (List[StandaloneObject], "List[dessia_common.forms.StandaloneObject]"),
        (List[List[int]], "List[List[int]]"),
        (List[Tuple[StandaloneObject, int]], "List[Tuple[dessia_common.forms.StandaloneObject, int]]"),
        (List[Tuple[int, Tuple[StandaloneObject, int], Distance]],
         "List[Tuple[int, Tuple[dessia_common.forms.StandaloneObject, int], dessia_common.measures.Distance]]"),

        # Dict
        (Dict[str, int], "Dict[str, int]"),
        (Dict[str, Tuple[int, float]], "Dict[str, Tuple[int, float]]"),

        # InstanceOf
        (InstanceOf[StandaloneObject], "InstanceOf[dessia_common.forms.StandaloneObject]"),

        # Subclass
        # (Subclass[StandaloneObject], "Subclass[dessia_common.forms.StandaloneObject]"),

        # Methods
        (MethodType[StandaloneObject], "MethodType[dessia_common.forms.StandaloneObject]"),
        (ClassMethodType[StandaloneObject], "ClassMethodType[dessia_common.forms.StandaloneObject]"),

        # Attributes
        (AttributeType[StandaloneObject], "AttributeType[dessia_common.forms.StandaloneObject]"),
        (ClassAttributeType[StandaloneObject], "ClassAttributeType[dessia_common.forms.StandaloneObject]"),

        # Types
        (Type, "Type"),
        (Type[StandaloneObject], "Type[dessia_common.forms.StandaloneObject]"),

        # Files
        (StringFile, "dessia_common.files.StringFile"),
        (BinaryFile, "dessia_common.files.BinaryFile")
    ])
    def test_annotation_serialization_process(self, annotation, expected_serialized):
        serialized = serialize_annotation(annotation)
        self.assertEqual(serialized, expected_serialized)

        deserialized = deserialize_annotation(serialized)
        self.assertEqual(deserialized, annotation)


if __name__ == '__main__':
    unittest.main(verbosity=2)
