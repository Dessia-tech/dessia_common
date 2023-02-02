#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Models for power simulation workflow. """
from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod
import dessia_common.typings as dct
import dessia_common.tests as dctest


instanciate_system = InstantiateModel(model_class=dctest.System, name='Instantiate Generator')

instanciate_system.outputs[0].memorize = True

simulate = ModelMethod(method_type=dct.MethodType(dctest.System, 'power_simulation'), name='Generator Generate')

pipe_1 = Pipe(input_variable=instanciate_system.outputs[0], output_variable=simulate.inputs[0])

blocks = [instanciate_system, simulate]
pipes = [pipe_1]
simulation_workflow = Workflow(blocks=blocks, pipes=pipes, output=simulate.outputs[0])
