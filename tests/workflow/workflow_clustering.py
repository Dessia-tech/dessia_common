"""
Cluster.py package testing for workflows
"""
import json
import pkg_resources
from dessia_common import tests
from dessia_common.core import HeterogeneousList
from dessia_common.cluster import CategorizedList
import dessia_common.workflow as wf

# =============================================================================
# CARS WITHOUT FEATURES
# =============================================================================
data_method = wf.MethodType(class_=tests.Car, name='from_csv')
block_data = wf.ClassMethod(method_type=data_method, name='data load')

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

categorized_list_method = wf.MethodType(class_=CategorizedList, name='from_dbscan')
block_cluster = wf.ClassMethod(method_type=categorized_list_method, name='labelling elements of list')

block_workflow = [block_data, block_heterogeneous_list, block_cluster]
pipe_worflow = [wf.Pipe(block_data.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_cluster.inputs[0])]
workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

workflow_run = workflow.run({
    workflow.index(block_data.inputs[0]): pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv'),
    workflow.index(block_cluster.inputs[1]): 40})

# Workflow tests
workflow._check_platform()
workflow.plot()
workflow.display_settings()
workflow_run.output_value.plot()

# JSON TESTS
dict_workflow = workflow.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow)
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)

# Workflow_run tests: do not run on local but run on platform
# dict_workflow_run = workflow_run.to_dict(use_pointers=True)
# json_dict = json.dumps(dict_workflow_run)
# decoded_json = json.loads(json_dict)
# deserialized_object = workflow_run.dict_to_object(decoded_json)

# =============================================================================
# CARS WITH FEATURES
# =============================================================================
data_method = wf.MethodType(class_=tests.CarWithFeatures, name='from_csv')
block_data = wf.ClassMethod(method_type=data_method, name='data load')

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

categorized_list_method = wf.MethodType(class_=CategorizedList, name='from_dbscan')
block_cluster = wf.ClassMethod(method_type=categorized_list_method, name='labelling elements of list')

block_workflow = [block_data, block_heterogeneous_list, block_cluster]
pipe_worflow = [wf.Pipe(block_data.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_cluster.inputs[0])]
workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

workflow_run = workflow.run({
    workflow.index(block_data.inputs[0]): pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv'),
    workflow.index(block_cluster.inputs[1]): 40})

# Workflow tests
workflow._check_platform()
workflow.plot()
workflow.display_settings()
workflow_run.output_value.plot()

# JSON TESTS
dict_workflow = workflow.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow)
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)

# Workflow_run tests: do not run on local but run on platform
# dict_workflow_run = workflow_run.to_dict(use_pointers=True)
# json_dict = json.dumps(dict_workflow_run)
# decoded_json = json.loads(json_dict)
# deserialized_object = workflow_run.dict_to_object(decoded_json)

# =============================================================================
# RANDDATA SMALL DATASET AND BLOC CONCATENATE
# =============================================================================
mean_borns = (-50, 50)
std_borns = (-2, 2)

data_method_5 = wf.MethodType(class_=tests.RandDataD5, name='create_dataset')
block_data_d5 = wf.ClassMethod(method_type=data_method_5, name='data d5')

data_method_4 = wf.MethodType(class_=tests.RandDataD4, name='create_dataset')
block_data_d4 = wf.ClassMethod(method_type=data_method_4, name='data d4')

data_method_3 = wf.MethodType(class_=tests.RandDataD3, name='create_dataset')
block_data_d3 = wf.ClassMethod(method_type=data_method_3, name='data d3')

block_concatenate = wf.Concatenate(3)

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

categorized_list_method = wf.MethodType(class_=CategorizedList, name='from_dbscan')
block_cluster = wf.ClassMethod(method_type=categorized_list_method, name='labelling elements of list')

block_workflow = [block_data_d5, block_data_d4, block_data_d3,
                  block_concatenate, block_heterogeneous_list, block_cluster]

pipe_worflow = [wf.Pipe(block_data_d5.outputs[0], block_concatenate.inputs[0]),
                wf.Pipe(block_data_d4.outputs[0], block_concatenate.inputs[1]),
                wf.Pipe(block_data_d3.outputs[0], block_concatenate.inputs[2]),
                wf.Pipe(block_concatenate.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_cluster.inputs[0])]

workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

workflow_run = workflow.run({workflow.index(block_data_d5.inputs[0]): 10, workflow.index(block_data_d5.inputs[1]): 500,
                             workflow.index(block_data_d5.inputs[2]): mean_borns,
                             workflow.index(block_data_d5.inputs[3]): std_borns,

                             workflow.index(block_data_d4.inputs[0]): 10, workflow.index(block_data_d4.inputs[1]): 500,
                             workflow.index(block_data_d4.inputs[2]): mean_borns,
                             workflow.index(block_data_d4.inputs[3]): std_borns,

                             workflow.index(block_data_d3.inputs[0]): 10, workflow.index(block_data_d3.inputs[1]): 500,
                             workflow.index(block_data_d3.inputs[2]): mean_borns,
                             workflow.index(block_data_d3.inputs[3]): std_borns,

                             workflow.index(block_cluster.inputs[1]): 5})

# Workflow tests
workflow._check_platform()
workflow.plot()
workflow.display_settings()
workflow_run.output_value.plot()

# JSON TESTS
dict_workflow = workflow.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow)
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)

# Workflow_run tests: do not run on local but run on platform
dict_workflow_run = workflow_run.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow_run)
decoded_json = json.loads(json_dict)
deserialized_object = workflow_run.dict_to_object(decoded_json)

# Debug of block display, kept for now, will be removed soon
# gg = workflow_run._display_from_selector('plot_data')
# json.dumps(workflow_run.to_dict())
# workflow_run._displays
