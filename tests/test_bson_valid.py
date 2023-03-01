#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:43:53 2021

@author: steven
"""

from dessia_common.core import is_bson_valid


import unittest
from parameterized import parameterized_class


@parameterized_class(('value', 'expected_result'), [
    ({'a': 4.3, 'b': [{3: 'a'}]}, True),
    ({'a.2': 4.3, 'b': [{3: 'a'}]}, False),
    ({'a': 4.3, 'b': [{3.3: 'a'}]}, False),
    ({'a': float, 'b': [{2: 'a'}]}, False),
])
class TestBsonValid(unittest.TestCase):

    def test_bson_valid(self):
        valid, hint = is_bson_valid(self.value, allow_nonstring_keys=True)
        self.assertEqual(valid, self.expected_result)
