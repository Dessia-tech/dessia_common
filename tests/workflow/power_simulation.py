#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


from dessia_common.models.workflows import simulation_workflow
from dessia_common.models.power_test import components, component_connections, usage


workflow_run = simulation_workflow.run({0: components, 1:component_connections, 3:usage})
print(workflow_run.log)

workflow_run._check_platform()


manual_run = simulation_workflow.start_run({0: components, 1:component_connections})

print(manual_run)

evaluated_block = manual_run.evaluate_next_block()
assert evaluated_block is not None
print(evaluated_block)

evaluated = manual_run.block_evaluation(1)
print(evaluated)
assert not evaluated

manual_run.add_input_value(3, usage)

evaluated_blocks = manual_run.continue_run()
assert(manual_run.progress == 1)
print(manual_run.progress)

manual_run._displays()