#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dessia_common.core import DessiaObject
from dessia_common.models.forms import standalone_object

import unittest
from parameterized import parameterized


class TestSerialization(unittest.TestCase):

    @parameterized.expand([
        (standalone_object,),
    ])
    def test_objects_serialization_deserialization(self, my_obj):
        d = my_obj.to_dict()
        obj = DessiaObject.dict_to_object(d)
        assert obj == my_obj


if __name__ == '__main__':
    unittest.main()
