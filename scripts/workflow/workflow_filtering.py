"""
Tests on filtering methods with block filter and Dataset.filtering
"""
import json
import pkg_resources
from dessia_common.files import StringFile
from dessia_common.typings import ClassMethodType, MethodType
from dessia_common.tests import Car
from dessia_common.workflow.blocks import ClassMethod, InstantiateModel, Filter, ModelMethod
from dessia_common.core import DessiaFilter, FiltersList
from dessia_common.datatools.dataset import Dataset
from dessia_common.workflow.core import Workflow, Pipe

# Import data
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
stream_file = StringFile.from_stream(csv_cars)

# =============================================================================================================
# Filters Workflow
# =============================================================================================================

block_0 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='CSV Cars')
block_1 = InstantiateModel(model_class=Dataset, name='HList Cars')
block_2 = ModelMethod(method_type=MethodType(Dataset, 'filtering'), name='Filters HList')
block_3 = ClassMethod(method_type=ClassMethodType(FiltersList, 'from_filters_list'), name='Filters Hlist')
block_4 = Filter(filters=[DessiaFilter(attribute='weight', comparison_operator='<=', bound=4000, name='weight'),
                          DessiaFilter(attribute='mpg', comparison_operator='>=', bound=25, name='mpg')],
                 logical_operator='xor', name='Filters on W or MPG')
blocks = [block_0, block_1, block_2, block_3, block_4]

pipe_0 = Pipe(block_1.outputs[0], block_2.inputs[0])
pipe_1 = Pipe(block_3.outputs[0], block_2.inputs[1])
pipe_2 = Pipe(block_0.outputs[0], block_4.inputs[0])
pipe_3 = Pipe(block_4.outputs[0], block_1.inputs[0])
pipes = [pipe_0, pipe_1, pipe_2, pipe_3]

workflow = Workflow(blocks, pipes, output=block_2.outputs[0], name='Filters demo')

# Workflow run
filters = [DessiaFilter('cylinders', ">", 4), DessiaFilter('displacement', ">", 0.25)]
workflow_run = workflow.run({
    workflow.index(block_0.inputs[0]): stream_file,
    workflow.index(block_3.inputs[0]): filters,
    workflow.index(block_3.inputs[1]): 'and'})

# Workflow tests
workflow._check_platform()

# JSON TESTS
output_dict = workflow_run.output_value[[0, 3, 9, 11, 25, 44]].to_dict(use_pointers=True)
output_json = json.dumps(output_dict)
output_json_to_dict = json.loads(output_json)
output_jsondict_to_object = Dataset.dict_to_object(output_json_to_dict)

reference_output = {
  "name": "",
  "object_class": "dessia_common.datatools.dataset.Dataset",
  "package_version": "0.9.2.dev320+gecd4609",
  "dessia_objects": [
    {
      "name": "Chevrolet Chevelle Malibu",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 18.0,
      "cylinders": 8.0,
      "displacement": 0.307,
      "horsepower": 130.0,
      "weight": 3504.0,
      "acceleration": 12.0,
      "model": 70.0,
      "origin": "US\r"
    },
    {
      "name": "AMC Rebel SST",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 16.0,
      "cylinders": 8.0,
      "displacement": 0.304,
      "horsepower": 150.0,
      "weight": 3433.0,
      "acceleration": 12.0,
      "model": 70.0,
      "origin": "US\r"
    },
    {
      "name": "Ford Mustang Boss 302",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 0.0,
      "cylinders": 8.0,
      "displacement": 0.302,
      "horsepower": 140.0,
      "weight": 3353.0,
      "acceleration": 8.0,
      "model": 70.0,
      "origin": "US\r"
    },
    {
      "name": "Buick Estate Wagon (sw)",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 14.0,
      "cylinders": 8.0,
      "displacement": 0.455,
      "horsepower": 225.0,
      "weight": 3086.0,
      "acceleration": 10.0,
      "model": 70.0,
      "origin": "US\r"
    },
    {
      "name": "AMC Matador",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 15.5,
      "cylinders": 8.0,
      "displacement": 0.304,
      "horsepower": 120.0,
      "weight": 3962.0,
      "acceleration": 13.9,
      "model": 76.0,
      "origin": "US\r"
    },
    {
      "name": "Oldsmobile Cutlass Salon Brougham",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 23.9,
      "cylinders": 8.0,
      "displacement": 0.26,
      "horsepower": 90.0,
      "weight": 3420.0,
      "acceleration": 22.2,
      "model": 79.0,
      "origin": "US\r"
    }]
}

# Test
assert(output_jsondict_to_object == Dataset.dict_to_object(reference_output))
