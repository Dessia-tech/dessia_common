#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:43:53 2021

@author: steven
"""

from dessia_common.utils.diff import dict_diff


ds1 = {'key': 'val1'}
ds2 = {'key': 'val2'}

path, val1, val2 = dict_diff(ds1, ds2)[0][0]
assert path == '#/key'
assert val1 == 'val1'
assert val2 == 'val2'
