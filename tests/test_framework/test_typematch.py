from dessia_common.utils.types import typematch
from dessia_common.core import DessiaObject, PhysicalObject
from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues
from typing import List, Tuple, Union, Any, Optional, Dict
from dessia_common.measures import Measure
import unittest
from parameterized import parameterized


class TestTypeMatches(unittest.TestCase):
    @parameterized.expand([
        # TRIVIAL AND SPECIFIC
        (DessiaObject, Any),
        (int, Any),
        (int, float),

        # INHERITANCE
        (DessiaObject, object),

        # Lists
        (List[int], List[int]),
        (List[Measure], List[float]),
        (List[StandaloneObjectWithDefaultValues], List[DessiaObject]),
        (List[StandaloneObjectWithDefaultValues], List[StandaloneObject]),

        # TUPLES
        (Tuple[int, str], Tuple[int, str]),

        # DICTS
        (Dict[str, PhysicalObject], Dict[str, DessiaObject]),

        # DEFAULT VALUES
        (Optional[List[StandaloneObject]], List[DessiaObject]),
        (List[StandaloneObject], Optional[List[DessiaObject]]),
        (Union[List[StandaloneObject], type(None)], List[DessiaObject]),

        # UNION
        (DessiaObject, Union[DessiaObject, int]),
        (StandaloneObjectWithDefaultValues, Union[DessiaObject, StandaloneObject]),
        (Union[str, int], Union[str, int]),
        (Union[str, int], Union[bool, int, str]),
        (Union[str, int], Union[int, str])
    ])
    def test_positive_matches(self, type_, match_against):
        self.assertTrue(typematch(type_, match_against))

    @parameterized.expand([
        # TRIVIAL AND SPECIFIC
        (float, int),

        # INHERITANCE
        (object, DessiaObject),

        # Lists
        (List[float], List[Measure]),

        # TUPLES
        (Tuple[int, int], Tuple[str, int]),

        # DICTS
        (Dict[str, str], Dict[int, str]),
        (Dict[str, str], Dict[str, int]),

        # UNION
        (DessiaObject, Union[str, int]),
        (Union[DessiaObject, int], DessiaObject),

        # UNEQUAL COMPLEX
        (List[int], Tuple[int])
    ])
    def test_negative_matches(self, type_, match_against):
        self.assertFalse(typematch(type_, match_against))
