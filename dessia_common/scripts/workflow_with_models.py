#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""

#import math
import dessia_common.workflow as workflow


class Generator:
    def __init__(self, parameter, nb_solutions=25):
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
        
    def optimize(self, optimization_value=3):
        self.model_to_optimize.value += optimization_value
    
instanciate_generator = workflow.InstanciateModel(Generator)
generator_generate = workflow.ModelMethod(Generator, 'generate')
attribute_selection = workflow.ModelAttribute('models')

# Subworkflow of model optimization
instanciate_optimizer = workflow.InstanciateModel(Optimizer)
optimization = workflow.ModelMethod(Optimizer, 'optimize')
model_fetcher = workflow.ModelAttribute('model_to_optimize')

pipe1_opt = workflow.Pipe(instanciate_optimizer.outputs[0], optimization.inputs[0])
pipe2_opt = workflow.Pipe(optimization.outputs[1], model_fetcher.inputs[0])
optimization_workflow = workflow.Workflow([instanciate_optimizer, optimization,
                                           model_fetcher],
                                          [pipe1_opt, pipe2_opt],
                                          model_fetcher.outputs[0])

parallel_optimization = workflow.ForEach(optimization_workflow, instanciate_optimizer.inputs[0])

pipe_1 = workflow.Pipe(instanciate_generator.outputs[0], generator_generate.inputs[0])
pipe_2 = workflow.Pipe(generator_generate.outputs[1], attribute_selection.inputs[0])
pipe_3 = workflow.Pipe(attribute_selection.outputs[0], parallel_optimization.inputs[0])


demo_workflow = workflow.Workflow([instanciate_generator,
                                   generator_generate,
                                   attribute_selection,
                                   parallel_optimization
                                   ],
                                 [pipe_1, pipe_2, pipe_3],
                                 parallel_optimization.outputs[0]
                                 )

demo_workflow.plot_graph()

input_values = {instanciate_generator.inputs[0]: 3.21,
#                instanciate_generator.inputs[1]: 25,
                }

demo_workflow_run = demo_workflow.run(input_values, verbose=True)


demo_workflow_dict = demo_workflow.to_dict()
import json
demo_workflow_json = json.dumps(demo_workflow_dict)
demo_workflow_dict_from_json = json.loads(demo_workflow_json)
deserialized_demo_workflow = workflow.Workflow.dict_to_object(demo_workflow_dict_from_json)
assert demo_workflow == deserialized_demo_workflow