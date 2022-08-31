"""
Test for pareto in workflows
"""

import pkg_resources
from dessia_common.files import StringFile
from dessia_common.typings import ClassMethodType, MethodType
from dessia_common.tests import Car
from dessia_common.workflow.blocks import ClassMethod, InstantiateModel, ModelMethod, Sequence
from dessia_common.datatools import HeterogeneousList
from dessia_common.workflow.core import Workflow, Pipe

# Import data
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
stream_file = StringFile.from_stream(csv_cars)

# Workflow
block_0 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='CSV Cars')
block_1 = InstantiateModel(model_class=HeterogeneousList, name='HeterogeneousList')
block_2 = ModelMethod(method_type=MethodType(HeterogeneousList, 'get_attribute_values'), name='Weight')
block_3 = ModelMethod(method_type=MethodType(HeterogeneousList, 'get_attribute_values'), name='MPG')
block_4 = Sequence(number_arguments=2, name='list of costs')
block_5 = ModelMethod(method_type=MethodType(HeterogeneousList, 'pareto_points'), name='Pareto Points')
blocks = [block_0, block_1, block_2, block_3, block_4, block_5]

pipe_0 = Pipe(block_0.outputs[0], block_1.inputs[0])
pipe_1 = Pipe(block_1.outputs[0], block_2.inputs[0])
pipe_2 = Pipe(block_1.outputs[0], block_3.inputs[0])
pipe_3 = Pipe(block_2.outputs[0], block_4.inputs[0])
pipe_4 = Pipe(block_3.outputs[0], block_4.inputs[1])
pipe_5 = Pipe(block_4.outputs[0], block_5.inputs[1])
pipe_6 = Pipe(block_1.outputs[0], block_5.inputs[0])
pipes = [pipe_0, pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6]

workflow = Workflow(blocks, pipes, output=block_5.outputs[0], name='Pick Pareto members')

# Workflow run
workflow_run = workflow.run({
    workflow.index(block_0.inputs[0]): stream_file,
    workflow.index(block_2.inputs[1]): 'weight',
    workflow.index(block_3.inputs[1]): 'mpg'})

# Workflow tests
workflow._check_platform()
workflow.plot()
workflow_run.output_value.plot()

