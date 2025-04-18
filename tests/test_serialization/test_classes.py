from dessia_common.serialization import deserialize
from dessia_common.utils.types import is_classname_transform
from dessia_common.core import DessiaObject
from dessia_common.workflow.core import Workflow
import unittest
from parameterized import parameterized


class TestStringDeserialization(unittest.TestCase):
    @parameterized.expand([
        ("", False),
        (".", False),
        (".suffix", False),
        ("prefix.suffix", False),
        ("prefix.", False),
        ("word", False),
        (" ", False),
        ("    ", False),
        ("  .  ", False),
        ("1.23", False),
        ("dessia_common.core.DessiaObject", DessiaObject),
        ("dessia_common.workflow.core.Workflow", Workflow),
        ("DessiaObject", False),
    ])
    def test_classname_transform(self, rawstring, expected_result):
        self.assertEqual(is_classname_transform(rawstring), expected_result)

    @parameterized.expand([
        ("", ""),
        (".", "."),
        (".suffix", ".suffix"),
        ("prefix.suffix", "prefix.suffix"),
        ("prefix.", "prefix."),
        ("word", "word"),
        (" ", " "),
        ("    ", "    "),
        ("  .  ", "  .  "),
        ("1.23", "1.23"),
        ("dessia_common.core.DessiaObject", DessiaObject),
        ("dessia_common.workflow.core.Workflow", Workflow),
        ("DessiaObject", "DessiaObject"),
    ])
    def test_string_deserialization(self, rawstring, expected_result):
        self.assertEqual(deserialize(rawstring), expected_result)