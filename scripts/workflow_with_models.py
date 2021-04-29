#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""

import dessia_common.workflow as wf
from dessia_common import DessiaObject


class Submodel(DessiaObject):
    _generic_eq = True

    def __init__(self, subvalue: int, name: str = ''):
        self.subvalue = subvalue
        self.name = name

        DessiaObject.__init__(self, name=name)


class Model(DessiaObject):
    _generic_eq = True

    def __init__(self, value: int, submodel: Submodel, name: str = ''):
        self.value = value
        self.submodel = submodel

        DessiaObject.__init__(self, name=name)


class Generator(DessiaObject):
    def __init__(self, parameter: int, nb_solutions: int = 25, name: str = ''):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        self.models = None

        DessiaObject.__init__(self, name=name)

    def generate(self) -> None:
        submodels = [Submodel(self.parameter * i)
                     for i in range(self.nb_solutions)]
        self.models = [Model(self.parameter + i, submodels[i])
                       for i in range(self.nb_solutions)]


class Optimizer(DessiaObject):
    def __init__(self, model_to_optimize: Model, name: str = ''):
        self.model_to_optimize = model_to_optimize

        DessiaObject.__init__(self, name=name)

    def optimize(self, optimization_value: int = 3) -> None:
        self.model_to_optimize.value += optimization_value


instanciate_generator = wf.InstanciateModel(model_class=Generator,
                                            name='Instantiate Generator')
generator_generate = wf.ModelMethod(model_class=Generator,
                                    method_name='generate',
                                    name='Generator Generate')
attribute_selection = wf.ModelAttribute(attribute_name='models',
                                        name='Attribute Selection')

# Subworkflow of model optimization
instanciate_optimizer = wf.InstanciateModel(model_class=Optimizer,
                                            name='Instantiate Optimizer')
optimization = wf.ModelMethod(model_class=Optimizer,
                              method_name='optimize',
                              name='Optimization')
model_fetcher = wf.ModelAttribute(attribute_name='model_to_optimize',
                                  name='Model Fetcher')

pipe1_opt = wf.Pipe(input_variable=instanciate_optimizer.outputs[0],
                    output_variable=optimization.inputs[0])
pipe2_opt = wf.Pipe(input_variable=optimization.outputs[1],
                    output_variable=model_fetcher.inputs[0])
optimization_blocks = [instanciate_optimizer, optimization, model_fetcher]
optimization_pipes = [pipe1_opt, pipe2_opt]
optimization_workflow = wf.Workflow(blocks=optimization_blocks,
                                    pipes=optimization_pipes,
                                    output=model_fetcher.outputs[0],
                                    name='Optimization Workflow')

optimization_workflow_block = wf.WorkflowBlock(workflow=optimization_workflow,
                                               name='Workflow Block')

parallel_optimization = wf.ForEach(
    workflow_block=optimization_workflow_block,
    workflow_iterable_input=optimization_workflow_block.inputs[0],
    name='ForEach'
)

unpacker = wf.Unpacker(indices=[0, 3, -1], name='Unpacker')
sequence = wf.Sequence(number_arguments=2, name='Sequence')

filters = [{'attribute': 'value', 'operator': 'gt', 'bound': 0},
           {'attribute': 'submodel.subvalue', 'operator': 'lt', 'bound': 200}]

filter_sort = wf.Filter(filters=filters, name='Filters')

pipe_1 = wf.Pipe(input_variable=instanciate_generator.outputs[0],
                 output_variable=generator_generate.inputs[0])
pipe_2 = wf.Pipe(input_variable=generator_generate.outputs[1],
                 output_variable=attribute_selection.inputs[0])
pipe_3 = wf.Pipe(input_variable=attribute_selection.outputs[0],
                 output_variable=parallel_optimization.inputs[0])
pipe_4 = wf.Pipe(input_variable=parallel_optimization.outputs[0],
                 output_variable=unpacker.inputs[0])
pipe_51 = wf.Pipe(input_variable=unpacker.outputs[0],
                  output_variable=sequence.inputs[0])
pipe_52 = wf.Pipe(input_variable=unpacker.outputs[2],
                  output_variable=sequence.inputs[1])
pipe_6 = wf.Pipe(input_variable=sequence.outputs[0],
                 output_variable=filter_sort.inputs[0])

blocks = [instanciate_generator, generator_generate, attribute_selection,
          parallel_optimization, unpacker, sequence, filter_sort]
pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_51, pipe_52, pipe_6]
demo_workflow = wf.Workflow(blocks=blocks, pipes=pipes,
                            output=unpacker.outputs[0])

# demo_workflow.plot_graph()

input_values = {0: 5}

demo_workflow_run = demo_workflow.run(input_values=input_values,
                                      verbose=True)

# Assert to_dict, dict_to_object, hashes, eqs
dict_ = demo_workflow_run.to_dict()
object_ = wf.WorkflowRun.dict_to_object(dict_=dict_)

assert hash(demo_workflow_run) == hash(object_)

# Assert deserialization
# demo_workflow_dict = demo_workflow.to_dict()
# import json
# demo_workflow_json = json.dumps(demo_workflow_dict)
# dict_from_json = json.loads(demo_workflow_json)
# deserialized_demo_workflow = wf.Workflow.dict_to_object(dict_from_json)
# assert demo_workflow == deserialized_demo_workflow
