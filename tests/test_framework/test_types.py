from dessia_common.utils.types import is_sequence, is_list, is_tuple, isinstance_base_types, is_simple
import unittest
from parameterized import parameterized


class TestSimpleTypes(unittest.TestCase):
    @parameterized.expand([
        ([], True),
        ((1,), True),
        ({}, False),
        (10, False)
    ])
    def test_is_sequence(self, value, expected_result):
        self.assertTrue(is_sequence(value) is expected_result)

    @parameterized.expand([
        ([], True),
        ((1,), False)
    ])
    def test_is_list(self, value, expected_result):
        self.assertTrue(is_list(value) is expected_result)

    @parameterized.expand([
        ([], False),
        ((1,), True)
    ])
    def test_is_tuple(self, value, expected_result):
        self.assertTrue(is_tuple(value) is expected_result)

    @parameterized.expand([
        ([], False),
        ((1,), False),
        ({}, False),
        (False, True),
        (3.12, True),
        (None, True),
        (3, True),
        ("3", True)
    ])
    def test_is_base_type(self, value, expected_result):
        self.assertTrue(isinstance_base_types(value) is expected_result)

    @parameterized.expand([
        (False, False),
        (3.12, False),
        (None, True),
        (3, True),
        ("3", True)
    ])
    def test_is_simple(self, value, expected_result):
        self.assertTrue(is_simple(value) is expected_result)
