from dessia_common.schemas.core import HomogeneousSequence, HeterogeneousSequence
from dessia_common.forms import StandaloneObject
from typing import List, Tuple

import unittest
from parameterized import parameterized


class TestFaulty(unittest.TestCase):
    @parameterized.expand([
        (HomogeneousSequence(annotation=List, attribute="bare_list"), 2),
        (HeterogeneousSequence(annotation=Tuple, attribute="bare_tuple"), 2)
    ])
    def test_schema_check(self, schema, expected_number):
        checked_schema = schema.check_list()
        errors = checked_schema.checks_above_level("error")
        self.assertTrue(errors)
        self.assertEqual(len(errors), expected_number)
        self.assertEqual(checked_schema[0].level, "error")


class TestSequences(unittest.TestCase):
    @parameterized.expand([
        (HomogeneousSequence(annotation=List[int], attribute="integer_list"), "List[int]"),
    ])
    def test_simple_sequences(self, schema, expected_typing):
        computed_schema = schema.to_dict("Sequence")
        self.assertEqual(computed_schema["type"], "array")
        self.assertEqual(computed_schema["title"], "Sequence")
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["items"]["type"], "number")
        self.assertEqual(computed_schema["items"]["python_typing"], "int")
        self.assertEqual(computed_schema["items"]["title"], "Sequence/0")
        self.assertEqual(computed_schema["items"]["editable"], False)
        self.assertEqual(computed_schema["items"]["description"], "")

        checked_schema = schema.check_list()
        self.assertFalse(checked_schema.checks_above_level("error"))

    @parameterized.expand([
        (HomogeneousSequence(annotation=List[List[StandaloneObject]], attribute="nested_list"),
         "List[List[dessia_common.forms.StandaloneObject]]"),
    ])
    def test_nested_sequences(self, schema, expected_typing):
        computed_schema = schema.to_dict("Nested Sequence")
        self.assertEqual(computed_schema["type"], "array")
        self.assertEqual(computed_schema["title"], "Nested Sequence")
        self.assertEqual(computed_schema["python_typing"], expected_typing)

        self.assertEqual(computed_schema["items"]["type"], "array")
        self.assertEqual(computed_schema["items"]["python_typing"], "List[dessia_common.forms.StandaloneObject]")
        self.assertEqual(computed_schema["items"]["title"], "Nested Sequence/0")
        self.assertEqual(computed_schema["items"]["editable"], False)
        self.assertEqual(computed_schema["items"]["description"], "")

        self.assertEqual(computed_schema["items"]["items"]["type"], "object")
        self.assertEqual(computed_schema["items"]["items"]["python_typing"], "dessia_common.forms.StandaloneObject")
        self.assertEqual(computed_schema["items"]["items"]["title"], "Nested Sequence/0/0")

        checked_schema = schema.check_list()
        self.assertFalse(checked_schema.checks_above_level("error"))

    @parameterized.expand([
        (HeterogeneousSequence(annotation=Tuple[int, str], attribute="two_element_tuple"), "Tuple[int, str]"),
    ])
    def test_simple_sequences(self, schema, expected_typing):
        computed_schema = schema.to_dict("Sequence")
        self.assertEqual(computed_schema["type"], "array")
        self.assertEqual(computed_schema["additionalItems"], False)
        self.assertEqual(computed_schema["title"], "Sequence")
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["items"][0]["type"], "number")
        self.assertEqual(computed_schema["items"][0]["python_typing"], "int")
        self.assertEqual(computed_schema["items"][0]["title"], "Sequence/0")
        self.assertEqual(computed_schema["items"][1]["type"], "string")
        self.assertEqual(computed_schema["items"][1]["python_typing"], "str")
        self.assertEqual(computed_schema["items"][1]["title"], "Sequence/1")

        checked_schema = schema.check_list()
        self.assertFalse(checked_schema.checks_above_level("error"))

        # TODO Should object be allowed here ?
        # more_arg_tuple = schema_chunk(annotation=Tuple[int, str, StandaloneObject], title="More Arg Tuple",
        #                               editable=True, description="Testing Tuple with several args")


if __name__ == '__main__':
    unittest.main(verbosity=2)
