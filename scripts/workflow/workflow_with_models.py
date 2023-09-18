#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""
import json
import dessia_common.typings as dct
from dessia_common.workflow.core import Pipe, Workflow, WorkflowRun
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod, ModelAttribute, WorkflowBlock, ForEach,\
    Unpacker, Sequence, Filter
from dessia_common.core import DessiaFilter

import dessia_common.tests as dctests

instanciate_generator = InstantiateModel(model_class=dctests.Generator, name='Instantiate Generator')
generator_generate = ModelMethod(dct.MethodType(dctests.Generator, 'generate'), name='Generator Generate')
attribute_selection = ModelAttribute(attribute_name='models', name='Attribute Selection')

# Subworkflow of model optimization
instanciate_optimizer = InstantiateModel(model_class=dctests.Optimizer, name='Instantiate Optimizer')
optimization = ModelMethod(dct.MethodType(dctests.Optimizer, 'optimize'), name='Optimization')
model_fetcher = ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')

pipe1_opt = Pipe(input_variable=instanciate_optimizer.outputs[0], output_variable=optimization.inputs[0])
pipe2_opt = Pipe(input_variable=optimization.outputs[1], output_variable=model_fetcher.inputs[0])
optimization_blocks = [instanciate_optimizer, optimization, model_fetcher]
optimization_pipes = [pipe1_opt, pipe2_opt]
optimization_workflow = Workflow(blocks=optimization_blocks, pipes=optimization_pipes,
                                 output=model_fetcher.outputs[0], name='Optimization Workflow')

optimization_workflow_block = WorkflowBlock(workflow=optimization_workflow, name='Workflow Block')

parallel_optimization = ForEach(workflow_block=optimization_workflow_block, iter_input_index=0, name='ForEach')

unpacker = Unpacker(indices=[0, 3, -1], name='Unpacker')
sequence = Sequence(number_arguments=2, name='Sequence')

filters = [DessiaFilter(attribute='value', comparison_operator='gt', bound=0),
           DessiaFilter(attribute='submodel/subvalue', comparison_operator='lt', bound=2000)]

filter_sort = Filter(filters=filters, name='Filters')

pipe_1 = Pipe(input_variable=instanciate_generator.outputs[0], output_variable=generator_generate.inputs[0])
pipe_2 = Pipe(input_variable=generator_generate.outputs[1], output_variable=attribute_selection.inputs[0])
pipe_3 = Pipe(input_variable=attribute_selection.outputs[0], output_variable=parallel_optimization.inputs[0])
pipe_4 = Pipe(input_variable=parallel_optimization.outputs[0], output_variable=unpacker.inputs[0])
pipe_51 = Pipe(input_variable=unpacker.outputs[0], output_variable=sequence.inputs[0])
pipe_52 = Pipe(input_variable=unpacker.outputs[2], output_variable=sequence.inputs[1])
pipe_6 = Pipe(input_variable=sequence.outputs[0], output_variable=filter_sort.inputs[0])

blocks = [instanciate_generator, generator_generate, attribute_selection,
          parallel_optimization, unpacker, sequence, filter_sort]
pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_51, pipe_52, pipe_6]
demo_workflow = Workflow(blocks=blocks, pipes=pipes, output=unpacker.outputs[0])

input_values = {0: 5}

demo_workflow_run = demo_workflow.run(input_values=input_values, verbose=True)

# Assert to_dict, dict_to_object, hashes, eqs
dict_ = demo_workflow_run.to_dict()
demo_workflow_run2 = WorkflowRun.dict_to_object(dict_=dict_)

assert hash(demo_workflow_run) == hash(demo_workflow_run2)

assert demo_workflow_run2 == demo_workflow_run


demo_workflow_run_copy = demo_workflow_run.copy()
assert demo_workflow_run == demo_workflow_run_copy

demo_workflow._check_platform()
demo_workflow_run._check_platform()

# Assert deserialization
demo_workflow_dict = demo_workflow.to_dict()

demo_workflow_json = json.dumps(demo_workflow_dict)
dict_from_json = json.loads(demo_workflow_json)
deserialized_demo_workflow = Workflow.dict_to_object(dict_from_json)
assert demo_workflow == deserialized_demo_workflow

# Worflow State
demo_workflow_state = demo_workflow.start_run({})
demo_workflow_state.add_input_value(0, 5)
demo_workflow_state.evaluate_next_block()

assert len(demo_workflow_state.values) == 1

copy_workflow_state = demo_workflow_state.copy()

assert copy_workflow_state == demo_workflow_state

demo_workflow_state._check_platform()

assert len(demo_workflow_run.values) == 0  # It has no export/display, so values should not be stored

print("workflow_with_models.py has passed")
