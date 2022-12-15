#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:03:59 2021

@author: steven
"""

from dessia_common.models import simulation_list, system1
from dessia_common.core import DessiaObject
import dessia_common.utils.serialization as dcus
from dessia_common.displays import draw_networkx_graph

d = simulation_list.to_dict()
simulation_list_2 = DessiaObject.dict_to_object(d)

assert simulation_list_2 == simulation_list

simulation_list_copy = simulation_list.copy()
assert simulation_list_copy == simulation_list
diff = simulation_list_copy._data_diff(simulation_list)
assert diff.is_empty()
# Let this print to test diff utils __repr__
print(diff)

simulation_list.jsonschema()

pointer_analysis = dcus.pointers_analysis(simulation_list)
pointer_graph = dcus.pointer_graph(d)
draw_networkx_graph(pointer_graph)

system1._check_platform()
system1.jsonschema()

system1.save_to_file('system1')
system1_lff = DessiaObject.load_from_file('system1.json')
assert system1_lff == system1

memo = {}
a, memo = dcus.serialize_with_pointers(system1)
assert memo
