#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import dessia_common.utils.interpolation as dc_inter

value = 3.4
objects = [1, 2, 3.2, 4.736, 5, 8]
istep = dc_inter.istep_from_value_on_list(objects, value)

inter_value = dc_inter.interpolate_from_istep(objects, istep)

assert value == inter_value
