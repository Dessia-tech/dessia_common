#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:43:53 2021

@author: steven
"""

import dessia_common
from dessia_common.models.tests import standalone_object


import unittest
from parameterized import parameterized_class


@parameterized_class(('value',), [
    (standalone_object, ),
])
class TestSerialization(unittest.TestCase):

    def test_serialization(self):
        d = self.value.to_dict()
        obj = dessia_common.DessiaObject.dict_to_object(d)
        assert obj == self.value


if __name__ == '__main__':
    unittest.main()
