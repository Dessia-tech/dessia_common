#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""

import math
import dessia_common.workflow as workflow




class Generator:
    def __init__(self, parameter, nb_solutions):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        
    def generate(self):
        self.models = [Model(self.parameter+i) for i in range(self.nb_solutions)]
#        print('self.models', self.models)


class Model:
    def __init__(self, value):
        self.value = value
        

class Optimizer:
    def __init__(self, model_to_optimize):
        self.model_to_optimize = model_to_optimize
        
    def optimize(self):
        self.model_to_optimize.value += 1000
    
instanciate_generator = workflow.InstanciateModel(Generator)
generator_generate = workflow.ModelMethod(Generator, 'generate')
attribute_selection = workflow.ModelAttribute('models')

# Subworkflow of model optimization
instanciate_optimizer = workflow.InstanciateModel(Optimizer)
optimization = workflow.ModelMethod(Optimizer, 'optimize')
model_fetcher = workflow.ModelAttribute('model_to_optimize')

pipe1_opt = workflow.Pipe(instanciate_optimizer.outputs[0], optimization.inputs[0])
pipe2_opt = workflow.Pipe(optimization.outputs[1], model_fetcher.inputs[0])
optimization_workflow = workflow.WorkFlow([instanciate_optimizer, optimization,
                                           model_fetcher],
                                          [pipe1_opt, pipe2_opt],
                                          model_fetcher.outputs[0])

parallel_optimization = workflow.ForEach(optimization_workflow)

pipe_1 = workflow.Pipe(instanciate_generator.outputs[0], generator_generate.inputs[0])
pipe_2 = workflow.Pipe(generator_generate.outputs[1], attribute_selection.inputs[0])
pipe_3 = workflow.Pipe(attribute_selection.outputs[0], parallel_optimization.inputs[0])


workflow = workflow.WorkFlow([instanciate_generator,
                              generator_generate,
                              attribute_selection,
                              parallel_optimization
                              ],
                             [pipe_1, pipe_2, pipe_3],
                             parallel_optimization.outputs[0]
                             )

workflow.plot_graph()

workflow_run = workflow.run([math.pi/3, 4])