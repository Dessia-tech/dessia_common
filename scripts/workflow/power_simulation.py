#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import json

from dessia_common import REF_MARKER
from dessia_common.models.workflows import simulation_workflow
from dessia_common.models.power_test import components1, component_connections1, usage1

simulation_workflow.to_dict(use_pointers=False)
input_values = {0: components1, 1: component_connections1, 3: usage1}

workflow_run = simulation_workflow.run(input_values)


# Testing that there is no pointer when use_pointers=False
d = workflow_run.to_dict(use_pointers=False)
s = json.dumps(d)
if REF_MARKER in s:
    ind_ref = s.index(REF_MARKER)
    print(s[ind_ref - 300:ind_ref + 500])
    raise ValueError('Pointer detected with use_pointers=False')

simulation_workflow.save_script_to_file('_simulation_workflow')

print("script power_simulation.py has passed")
