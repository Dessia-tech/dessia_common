#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 08:07:43 2021

@author: steven
"""

import dessia_common.utils.algebra as dc_alg

print(dc_alg.number2factor(12))
print(dc_alg.number3factor(12))

print(dc_alg.get_incomplete_factors(12, dc_alg.number2factor(6)[0]))
