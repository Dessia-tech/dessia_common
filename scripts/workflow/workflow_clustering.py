"""
Test for ClusteredDataset in workflow. Test filtering method
"""
import json
import pkg_resources
from dessia_common.files import StringFile
from dessia_common.typings import ClassMethodType, MethodType
from dessia_common.tests import Car
from dessia_common.workflow.blocks import ClassMethod, InstantiateModel, ModelMethod, Unpacker, Concatenate
from dessia_common.core import DessiaFilter, FiltersList
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.cluster import ClusteredDataset
from dessia_common.workflow.core import Workflow, Pipe

# Import data
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
stream_file = StringFile.from_stream(csv_cars)

# ===============================================================================================================
# ClusteredDataset Workflow
# ===============================================================================================================

block_0 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='CSV Cars')
block_1 = InstantiateModel(model_class=Dataset, name='HList Cars')
block_2 = ClassMethod(method_type=ClassMethodType(FiltersList, 'from_filters_list'), name='Filters Clist')
block_3 = ClassMethod(method_type=ClassMethodType(ClusteredDataset, 'from_agglomerative_clustering'), name='Clustering')
block_4 = ModelMethod(method_type=MethodType(ClusteredDataset, 'filtering'), name='CList.filtering')
blocks = [block_0, block_1, block_2, block_3, block_4]

pipe_0 = Pipe(block_0.outputs[0], block_1.inputs[0])
pipe_1 = Pipe(block_1.outputs[0], block_3.inputs[0])
pipe_2 = Pipe(block_3.outputs[0], block_4.inputs[0])
pipe_3 = Pipe(block_2.outputs[0], block_4.inputs[1])
pipes = [pipe_0, pipe_1, pipe_2, pipe_3]

workflow = Workflow(blocks, pipes, output=block_4.outputs[0], name='Filters demo Clists')

# Workflow run
filters = [DessiaFilter('cylinders', ">", 6), DessiaFilter('displacement', ">", 0.3)]
workflow_run = workflow.run({workflow.input_index(block_0.inputs[0]): stream_file,
                             workflow.input_index(block_2.inputs[0]): filters,
                             workflow.input_index(block_2.inputs[1]): 'and',
                             workflow.input_index(block_3.inputs[1]): 10})

# Workflow tests
workflow._check_platform()
wfrun_plot_data = workflow_run.output_value.plot_data()

# JSON TESTS
output_dict = workflow_run.output_value[[0, 3, 10, 15, 30, -1]].to_dict(use_pointers=True)
output_json = json.dumps(output_dict)
output_json_to_dict = json.loads(output_json)
output_jsondict_to_object_1 = ClusteredDataset.dict_to_object(output_json_to_dict)

# ===============================================================================================================
# ClusteredDataset Big Workflow
# ===============================================================================================================

block_0 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='CSV Cars')
block_1 = InstantiateModel(model_class=Dataset, name='HList Cars')
block_2 = ClassMethod(method_type=ClassMethodType(ClusteredDataset, 'from_agglomerative_clustering'), name='Clustering')
block_3 = ModelMethod(method_type=MethodType(ClusteredDataset, 'clustered_sublists'), name='Sublists')
block_4 = Unpacker(indices=[3, 8, 9], name='Unpack_3_8_9')
block_5 = Concatenate(number_arguments=3, name='Concatenate')
block_6 = ClassMethod(method_type=ClassMethodType(ClusteredDataset, 'from_dbscan'), name='DBSCAN')
blocks = [block_0, block_1, block_2, block_3, block_4, block_5, block_6]

pipe_0 = Pipe(block_0.outputs[0], block_1.inputs[0])
pipe_1 = Pipe(block_1.outputs[0], block_2.inputs[0])
pipe_2 = Pipe(block_2.outputs[0], block_3.inputs[0])
pipe_3 = Pipe(block_3.outputs[0], block_4.inputs[0])
pipe_4 = Pipe(block_4.outputs[0], block_5.inputs[0])
pipe_5 = Pipe(block_4.outputs[1], block_5.inputs[1])
pipe_6 = Pipe(block_4.outputs[2], block_5.inputs[2])
pipe_7 = Pipe(block_5.outputs[0], block_6.inputs[0])
pipes = [pipe_0, pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7]

workflow = Workflow(blocks, pipes, output=block_6.outputs[0], name='SubClist + Concatenate')

# Workflow run
workflow_run = workflow.run({workflow.input_index(block_0.inputs[0]): stream_file,
                             workflow.input_index(block_2.inputs[1]): 10,
                             workflow.input_index(block_6.inputs[1]): 40})

# Workflow tests
workflow._check_platform()
wfrun_plot_data = workflow_run.output_value.plot_data()

# JSON TESTS
output_dict = workflow_run.output_value[[0, 3, 5, 2, 12, -1]].to_dict(use_pointers=True)
output_json = json.dumps(output_dict)
output_json_to_dict = json.loads(output_json)
output_jsondict_to_object_2 = ClusteredDataset.dict_to_object(output_json_to_dict)
print(output_jsondict_to_object_2)

# ===============================================================================================================
# TESTS ON RESULTS
# ===============================================================================================================

reference_output_1 = {
  "name": "",
  "object_class": "dessia_common.datatools.cluster.ClusteredDataset",
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
      "name": "Chevrolet Chevelle Concours (sw)",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 0.0,
      "cylinders": 8.0,
      "displacement": 0.35,
      "horsepower": 165.0,
      "weight": 4142.0,
      "acceleration": 11.5,
      "model": 70.0,
      "origin": "US\r"
    },
    {
      "name": "Plymouth 'Cuda 340",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 14.0,
      "cylinders": 8.0,
      "displacement": 0.34,
      "horsepower": 160.0,
      "weight": 3609.0,
      "acceleration": 8.0,
      "model": 70.0,
      "origin": "US\r"
    },
    {
      "name": "Chevrolet Impala",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 13.0,
      "cylinders": 8.0,
      "displacement": 0.35,
      "horsepower": 165.0,
      "weight": 4274.0,
      "acceleration": 12.0,
      "model": 72.0,
      "origin": "US\r"
    },
    {
      "name": "Oldsmobile Cutlass LS",
      "object_class": "dessia_common.tests.Car",
      "package_version": "0.9.2.dev320+gecd4609",
      "mpg": 26.6,
      "cylinders": 8.0,
      "displacement": 0.35,
      "horsepower": 105.0,
      "weight": 3725.0,
      "acceleration": 19.0,
      "model": 81.0,
      "origin": "US\r"
    }],
  "labels": [1, 0, 4, 1, 4, 1]}

reference_output_2 = {
  'name': '',
  'object_class': 'dessia_common.datatools.cluster.ClusteredDataset',
  'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
  'dessia_objects': [
    {
      'name': 'Volkswagen 1131 Deluxe Sedan',
      'object_class': 'dessia_common.tests.Car',
      'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
      'mpg': 26.0,
      'cylinders': 4.0,
      'displacement': 0.097,
      'horsepower': 46.0,
      'weight': 1835.0,
      'acceleration': 20.5,
      'model': 70.0,
      'origin': 'Europe\r'
    },
    {
     'name': 'Volkswagen Super Beetle 117',
     'object_class': 'dessia_common.tests.Car',
     'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
     'mpg': 0.0,
     'cylinders': 4.0,
     'displacement': 0.097,
     'horsepower': 48.0,
     'weight': 1978.0,
     'acceleration': 20.0,
     'model': 71.0,
     'origin': 'Europe\r'},
  {
     'name': 'Peugeot 304',
     'object_class': 'dessia_common.tests.Car',
     'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
     'mpg': 30.0,
     'cylinders': 4.0,
     'displacement': 0.079,
     'horsepower': 70.0,
     'weight': 2074.0,
     'acceleration': 19.5,
     'model': 71.0,
     'origin': 'Europe\r'},
  {
     'name': 'Toyota Corolla 1200',
     'object_class': 'dessia_common.tests.Car',
     'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
     'mpg': 31.0,
     'cylinders': 4.0,
     'displacement': 0.071,
     'horsepower': 65.0,
     'weight': 1773.0,
     'acceleration': 19.0,
     'model': 71.0,
     'origin': 'Japan\r'},
  {
     'name': 'Toyota Corolla 1200',
     'object_class': 'dessia_common.tests.Car',
     'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
     'mpg': 32.0,
     'cylinders': 4.0,
     'displacement': 0.071,
     'horsepower': 65.0,
     'weight': 1836.0,
     'acceleration': 21.0,
     'model': 74.0,
     'origin': 'Japan\r'},
  {
     'name': 'Chevy S-10',
     'object_class': 'dessia_common.tests.Car',
     'package_version': '0.8.1.dev105+gb6cb82bfeatcluster',
     'mpg': 31.0,
     'cylinders': 4.0,
     'displacement': 0.119,
     'horsepower': 82.0,
     'weight': 2720.0,
     'acceleration': 19.4,
     'model': 82.0,
     'origin': 'US\r'
     }],
  'labels': [0, 1, 1, 0, 0, 4]}

# Tests
assert(output_jsondict_to_object_1 == ClusteredDataset.dict_to_object(reference_output_1))
assert(output_jsondict_to_object_2 == ClusteredDataset.dict_to_object(reference_output_2))

