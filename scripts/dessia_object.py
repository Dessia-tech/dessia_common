#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test abilities of DessiaObject class
"""

import dessia_common


class Model(dessia_common.DessiaObject):
    def __init__(self, a, b, name=''):
        dessia_common.DessiaObject.__init__(self, a=a, b=b, name=name)


model = Model(2, 3.4, 'name of model')

print(model.to_dict())
