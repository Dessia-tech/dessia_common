#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:43:53 2021

@author: steven
"""

from dessia_common import is_bson_valid


import unittest

class TestBsonValid(unittest.TestCase):

    def test_bson_valid(self):
        valid_bson_dict = {'a': 4.3,
                           'b': [{3:'a'}]}
        valid, hint = is_bson_valid(valid_bson_dict)
        self.assertTrue(valid)

        invalid_bson_dict1 = {'a.2': 4.3,
                              'b': [{3:'a'}]}
        valid, hint = is_bson_valid(invalid_bson_dict1)
        self.assertFalse(valid)


        invalid_bson_dict2 = {'a': 4.3,
                              'b': [{3.3:'a'}]}
        valid, hint = is_bson_valid(invalid_bson_dict2)
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()