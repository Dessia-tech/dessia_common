#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:03:59 2021

@author: steven
"""

from dessia_common.models import simulation_list
import dessia_common as dc

d = simulation_list.to_dict()
simulation_list_2 = dc.DessiaObject.dict_to_object(d)

assert simulation_list_2 == simulation_list