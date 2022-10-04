#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:35:52 2022

@author: steven
"""

import dessia_common.breakdown as bd
from dessia_common.models import simulation1

assert bd.attrmethod_getter(simulation1, 'system.output_power(1)') == 0.8
