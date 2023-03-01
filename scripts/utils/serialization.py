#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:43:53 2021

@author: steven
"""

from dessia_common.utils.diff import dict_diff


ds1 = {'key': 'val1'}
ds2 = {'key': 'val2'}

dv1 = dict_diff(ds1, ds2).different_values[0]
assert dv1.path == '#/key'
assert dv1.value1 == 'val1'
assert dv1.value2 == 'val2'
