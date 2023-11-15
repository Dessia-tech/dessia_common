from dessia_common.schemas.core import DynamicDict, SchemaAttribute
from typing import Dict
from dessia_common.forms import StandaloneObject

import unittest
from parameterized import parameterized

INT_ATTRIBUTE = SchemaAttribute(name="wrong_key_type_dict")
NESTED_ATTRIBUTE = SchemaAttribute(name="nested_dict")


class TestFaulty(unittest.TestCase):
    @parameterized.expand([
        (DynamicDict(annotation=Dict[int, int], attribute=INT_ATTRIBUTE), 1),
        (DynamicDict(annotation=Dict[str, Dict[str, StandaloneObject]], attribute=NESTED_ATTRIBUTE), 1)
    ])
    def test_schema_check(self, schema, expected_number):
        checked_schema = schema.check_list()
        errors = checked_schema.checks_above_level("error")
        self.assertTrue(errors)
        self.assertEqual(len(errors), expected_number)
        self.assertEqual(errors[0].level, "error")


DICTIONARY = SchemaAttribute(name="dictionary")


class TestDicts(unittest.TestCase):
    @parameterized.expand([
        (DynamicDict(annotation=Dict[str, int], attribute=DICT_ATTRIBUTE), "Dict[str, int]"),
    ])
    def test_simple_dicts(self, schema, expected_typing):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], "object")
        self.assertEqual(computed_schema["title"], "Dictionary")
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["patternProperties"][".*"]["type"], "number")

        checked_schema = schema.check_list()
        self.assertFalse(checked_schema.checks_above_level("error"))


if __name__ == '__main__':
    unittest.main(verbosity=2)
