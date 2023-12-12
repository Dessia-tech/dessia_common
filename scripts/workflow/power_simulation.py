#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import json
from dessia_common.core import DessiaObject

from dessia_common import REF_MARKER
from dessia_common.models.workflows import simulation_workflow
from dessia_common.models.power_test import components1, component_connections1, usage1
from dessia_common.serialization import serialize

simulation_workflow.to_dict(use_pointers=False)
input_values = {0: components1,
                1: component_connections1,
                3: usage1}

workflow_run = simulation_workflow.run(input_values)

print(workflow_run.log)

arguments = {str(k): serialize(v) for k, v in input_values.items()}
arguments = simulation_workflow.dict_to_arguments(arguments, 'run')

workflow_run2 = DessiaObject.dict_to_object(workflow_run.to_dict())
workflow_run._check_platform()
workflow_run.to_dict(use_pointers=False)

manual_run = simulation_workflow.start_run({0: components1, 1: component_connections1})

# print(manual_run)

evaluated_block = manual_run.evaluate_next_block()
assert evaluated_block is not None
# print(evaluated_block)

evaluated = manual_run.block_evaluation(1)
# print(evaluated)
assert not evaluated

manual_run.add_input_value(3, usage1)

evaluated_blocks = manual_run.continue_run()
assert(manual_run.progress == 1)
print('progress: ', manual_run.progress)

manual_run._displays()
manual_run.to_dict(use_pointers=False)

manual_run.schema()
manual_run.performance_analysis()

# Testing that there is no pointer when use_pointers=False
d = workflow_run.to_dict(use_pointers=False)
s = json.dumps(d)
if REF_MARKER in s:
    ind_ref = s.index(REF_MARKER)
    print(s[ind_ref - 300:ind_ref + 500])
    raise ValueError('Pointer detected with use_pointers=False')

simulation_workflow.save_script_to_file('_simulation_workflow')

print("script power_simulation.py has passed")
