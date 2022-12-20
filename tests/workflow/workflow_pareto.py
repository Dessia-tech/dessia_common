"""
Test for pareto in workflows
"""
import json
import pkg_resources
from dessia_common.files import StringFile
from dessia_common.typings import ClassMethodType, MethodType
from dessia_common.tests import Car
from dessia_common.workflow.blocks import ClassMethod, InstantiateModel, ModelMethod
from dessia_common.datatools.dataset import Dataset
from dessia_common.workflow.core import Workflow, Pipe

# Import data
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
stream_file = StringFile.from_stream(csv_cars)

# Workflow
block_0 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='CSV Cars')
block_1 = InstantiateModel(model_class=Dataset, name='Dataset')
block_2 = ModelMethod(method_type=MethodType(Dataset, 'pareto_points'), name='Pareto Points')
blocks = [block_0, block_1, block_2]

pipe_0 = Pipe(block_0.outputs[0], block_1.inputs[0])
pipe_2 = Pipe(block_1.outputs[0], block_2.inputs[0])
pipes = [pipe_0, pipe_2]

workflow = Workflow(blocks, pipes, output=block_2.outputs[0], name='Pick Pareto members')

# Workflow run
workflow_run = workflow.run({
    workflow.index(block_0.inputs[0]): stream_file,
    workflow.index(block_2.inputs[1]): ['weight', 'mpg']})

# Workflow tests
json_dict = json.dumps(workflow.to_dict())
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)
workflow._check_platform()

workflow._check_platform()
wfrun_plot_data = workflow_run.output_value.plot_data()
# assert(json.dumps(wfrun_plot_data[0].to_dict())[50:100] == 's": [{"mpg": 26.0, "cylinders": 4.0, "displacement')
# assert(json.dumps(wfrun_plot_data[1].to_dict())[10400:10450] == ': ["mpg", "cylinders", "displacement", "horsepower')
# assert(json.dumps(wfrun_plot_data[2].to_dict())[50:100] == 'te_names": ["Index of reduced basis vector", "Sing')
