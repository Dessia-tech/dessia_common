from typing import Optional, List, Union, Dict, Type
from dessia_common.schemas.core import SchemaAttribute, get_schema
from dessia_common.measures import Distance
from dessia_common.core import DessiaObject
from dessia_common.files import BinaryFile, StringFile
from dessia_common.utils.types import MethodType

import unittest
from parameterized import parameterized

ATTRIBUTE = SchemaAttribute("dummy")


class TestFileRelevance(unittest.TestCase):
    @parameterized.expand([
        (BinaryFile, True),
        (StringFile, True),
        (Union[BinaryFile, StringFile], True),
        (List[Union[BinaryFile, StringFile]], True),
        (List[Union[BinaryFile, int]], True),
        (Optional[List[BinaryFile]], True),
        (Optional[List[Union[BinaryFile, StringFile, int]]], True),
        (Dict[str, BinaryFile], True),
        (Dict[str, List[Union[BinaryFile, StringFile]]], True),
        (int, False),
        (bool, False),
        (str, False),
        (Distance, False),
        (DessiaObject, False),
        (List[DessiaObject], False),
        (Dict[str, List[DessiaObject]], False),
        (MethodType[Type], False),
        (MethodType[DessiaObject], False),
        (Type, False),
        (Type[DessiaObject], False)
    ])
    def test_annotations(self, annotation, expected_file_relevance: bool):
        schema = get_schema(annotation=annotation, attribute=ATTRIBUTE)
        self.assertEqual(schema.is_file_related, expected_file_relevance)
