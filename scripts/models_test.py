#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
from dessia_common.models import simulation_list, system1
from dessia_common.core import DessiaObject
import dessia_common.serialization as dcs
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

simulation_list.schema()

pointer_analysis = dcs.pointers_analysis(simulation_list)
pointer_graph = dcs.pointer_graph(d)
draw_networkx_graph(pointer_graph)

system1._check_platform()
system1.schema()
system1.save_export_to_file('xlsx', 'generic_xlsx')
os.path.isfile('generic_xlsx.xlsx')

check_list = system1.check_list()

system1.save_to_file('system1')
system1_lff = DessiaObject.from_json('system1.json')
assert system1_lff == system1
