#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple workflow composed of functions
"""
import builtins

import dessia_common.typings as dct
import dessia_common.workflow as wf
from dessia_common import DessiaFilter

import dessia_common.tests as dctests

instanciate_generator = wf.InstantiateModel(model_class=dctests.Generator, name='Instantiate Generator')
generator_generate = wf.ModelMethod(dct.MethodType(dctests.Generator, 'generate'), name='Generator Generate')
attribute_selection = wf.ModelAttribute(attribute_name='models', name='Attribute Selection')

# Subworkflow of model optimization
instanciate_optimizer = wf.InstantiateModel(model_class=dctests.Optimizer, name='Instantiate Optimizer')
optimization = wf.ModelMethod(dct.MethodType(dctests.Optimizer, 'optimize'), name='Optimization')
model_fetcher = wf.ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')

pipe1_opt = wf.Pipe(input_variable=instanciate_optimizer.outputs[0], output_variable=optimization.inputs[0])
pipe2_opt = wf.Pipe(input_variable=optimization.outputs[1], output_variable=model_fetcher.inputs[0])
optimization_blocks = [instanciate_optimizer, optimization, model_fetcher]
optimization_pipes = [pipe1_opt, pipe2_opt]
optimization_workflow = wf.Workflow(blocks=optimization_blocks, pipes=optimization_pipes,
                                    output=model_fetcher.outputs[0], name='Optimization Workflow')

optimization_workflow_block = wf.WorkflowBlock(workflow=optimization_workflow, name='Workflow Block')

parallel_optimization = wf.ForEach(workflow_block=optimization_workflow_block, iter_input_index=0, name='ForEach')

unpacker = wf.Unpacker(indices=[0, 3, -1], name='Unpacker')
sequence = wf.Sequence(number_arguments=2, name='Sequence')

filters = [DessiaFilter(attribute='value', operator='gt', bound=0),
           DessiaFilter(attribute='submodel/subvalue', operator='lt', bound=2000)]

filter_sort = wf.Filter(filters=filters, name='Filters')

pipe_1 = wf.Pipe(input_variable=instanciate_generator.outputs[0], output_variable=generator_generate.inputs[0])
pipe_2 = wf.Pipe(input_variable=generator_generate.outputs[1], output_variable=attribute_selection.inputs[0])
pipe_3 = wf.Pipe(input_variable=attribute_selection.outputs[0], output_variable=parallel_optimization.inputs[0])
pipe_4 = wf.Pipe(input_variable=parallel_optimization.outputs[0], output_variable=unpacker.inputs[0])
pipe_51 = wf.Pipe(input_variable=unpacker.outputs[0], output_variable=sequence.inputs[0])
pipe_52 = wf.Pipe(input_variable=unpacker.outputs[2], output_variable=sequence.inputs[1])
pipe_6 = wf.Pipe(input_variable=sequence.outputs[0], output_variable=filter_sort.inputs[0])

blocks = [instanciate_generator, generator_generate, attribute_selection,
          parallel_optimization, unpacker, sequence, filter_sort]
pipes = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_51, pipe_52, pipe_6]
demo_workflow = wf.Workflow(blocks=blocks, pipes=pipes, output=unpacker.outputs[0])

input_values = {0: 5}

demo_workflow_run = demo_workflow.run(input_values=input_values, verbose=True)

# Assert to_dict, dict_to_object, hashes, eqs
dict_ = demo_workflow_run.to_dict()
demo_workflow_run2 = wf.WorkflowRun.dict_to_object(dict_=dict_)

assert hash(demo_workflow_run) == hash(demo_workflow_run2)

assert demo_workflow_run2 == demo_workflow_run


demo_workflow_run_copy = demo_workflow_run.copy()
assert demo_workflow_run == demo_workflow_run_copy

demo_workflow._check_platform()
# demo_workflow_run._check_platform()

# Assert deserialization
demo_workflow_dict = demo_workflow.to_dict()
import json
demo_workflow_json = json.dumps(demo_workflow_dict)
dict_from_json = json.loads(demo_workflow_json)
deserialized_demo_workflow = wf.Workflow.dict_to_object(dict_from_json)
assert demo_workflow == deserialized_demo_workflow


# Testing to_script
demo_workflow.to_script()

test_blocks = [
    wf.ForEach(optimization_workflow_block, 0),
    wf.Archive(),
    wf.ClassMethod(
        method_type=dct.ClassMethodType(dctests.Car, 'from_csv'),
        name='car_from_csv'
    ),
    wf.InstantiateModel(dctests.Car, name='Instantiate Car'),
    wf.ModelAttribute('model_to_optimize', name='Model Fetcher'),
    wf.ModelMethod(
        method_type=dct.MethodType(dctests.Car, 'to_vector'),
        name='car_to_vector'
    ),
    wf.Sequence(3, "sequence_name"),
    wf.SetModelAttribute('name', 'name_Name'),
    wf.Substraction("substraction_name"),
    wf.Sum(name="sum_name"),
    wf.Flatten('flatten_name'),
    wf.Filter([DessiaFilter("attributeFilter", "operatorFilter", bound=3.1415)]),
    wf.Unpacker([1, 3], "unpacker_name"),
    wf.Display(),
    wf.MultiPlot(['multiplot0', 'multiplot1']),
    wf.Product(4, "product_name"),
    wf.Export(
        method_type=dct.MethodType(dctests.Car, 'to_vector'),
        name='Export',
        export_name="export_name"),
]

test_pipes = [wf.Pipe(test_blocks[0].outputs[0], test_blocks[3].inputs[0])]

workflow_script = wf.Workflow(test_blocks, test_pipes, test_blocks[0].outputs[0], name="script_workflow")
workflow_script.to_script()
