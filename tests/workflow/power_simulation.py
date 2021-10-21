#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:07:02 2021

@author: masfaraud
"""

from dessia_common.models.workflows import simulation_workflow
from dessia_common.models.power_test import components, component_connections, usage


workflow_run = simulation_workflow.run({0: components, 1:component_connections, 3:usage})
print(workflow_run.log)

workflow_run._check_platform()


manual_run = simulation_workflow.start_run({0: components, 1:component_connections})

print(manual_run)

manual_run.evaluate_next_block()

manual_run.block_evaluation(simulation_workflow.blocks[1])

manual_run.continue_run()
print(manual_run.progress)

