#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:25:40 2021

@author: masfaraud
"""
import dessia_common.workflow as dcw
import dessia_common.typings as dct
import dessia_common.tests as dctest


instanciate_system = dcw.InstantiateModel(model_class=dctest.System, name='Instantiate Generator')

instanciate_system.outputs[0].memorize = True

simulate = dcw.ModelMethod(method_type=dct.MethodType(dctest.System, 'power_simulation'),
                           name='Generator Generate')

pipe_1 = dcw.Pipe(input_variable=instanciate_system.outputs[0], output_variable=simulate.inputs[0])

blocks = [instanciate_system, simulate]
pipes = [pipe_1]
simulation_workflow = dcw.Workflow(blocks=blocks, pipes=pipes, output=simulate.outputs[0])
