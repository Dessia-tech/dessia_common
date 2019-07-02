#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""

import math
import dessia_common.workflow as workflow




class Model1:
    def __init__(self, attribute1, attribute2):
        self.attribute1 = attribute1
        self.attribute2 = attribute2
        
    def model1_method1(self):
        return math.hypot(self.attribute1 + self.attribute2)

    def model1_method2(self):
        return math.hypot(self.attribute1 - self.attribute2)

        
class Model2:
    def __init__(self, attribute1, attribute2):
        self.attribute1 = attribute1
        self.attribute2 = attribute2

    def model2_method1(self):
        return math.sqrt(self.attribute1 + self.attribute2)

    def model2_method2(self):
        return math.sqrt(self.attribute1 - self.attribute2)


#pipe1 = workflow.Pipe(sinus.output, arcsin.input_args[0])

model1 = workflow.InstanciateModel(Model1)
model2 = workflow.InstanciateModel(Model2)

model1_method1 = workflow.ModelMethod(model1, 'model1_method1')
model1_method2 = workflow.ModelMethod(model1, 'model1_method2')
model2_method1 = workflow.ModelMethod(model2, 'model2_method1')
model2_method2 = workflow.ModelMethod(model2, 'model2_method2')

instanciate = workflow.InstanciateModel(Model1)
pipe = workflow.Pipe(instanciate.outputs[0], model1_method2.inputs[0])

workflow = workflow.WorkFlow([instanciate,
                              model1_method1,
#                              model1_method2,
#                              model2_method1,
#                              model2_method2
                              ],
                             [])

workflow.plot_graph()

workflow_run = workflow.run([math.pi/3])