from dessia_common.schemas.core import DynamicDict
from typing import Dict
from dessia_common.forms import StandaloneObject

import unittest
from parameterized import parameterized


class TestFaulty(unittest.TestCase):
    @parameterized.expand([
        (DynamicDict(annotation=Dict[int, int], attribute="wrong_key_type_dict"), 1),
        (DynamicDict(annotation=Dict[str, Dict[str, StandaloneObject]], attribute="nested_dict"), 1)
    ])
    def test_schema_check(self, schema, expected_number):
        # computed_schema = schema.to_dict()
        # self.assertEqual(computed_schema["type"], "array")
        # self.assertEqual(computed_schema["items"], [])

        checked_schema = schema.check_list()
        self.assertTrue(checked_schema.checks_above_level("error"))
        self.assertEqual(len(checked_schema), expected_number)
        self.assertEqual(checked_schema[0].level, "error")


class TestDicts(unittest.TestCase):
    @parameterized.expand([
        (DynamicDict(annotation=Dict[str, int], attribute="dictionnary"), "Dict[str, int]"),
    ])
    def test_simple_dicts(self, schema, expected_typing):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], "object")
        self.assertEqual(computed_schema["title"], "Sequence")
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["patternProperties"][".*"]["type"], "number")

        checked_schema = schema.check_list()
        self.assertFalse(checked_schema.checks_above_level("error"))