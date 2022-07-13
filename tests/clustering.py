"""
Cluster.py package testing.
"""
import json
import pkg_resources
from dessia_common import tests, cluster
from dessia_common.core import HeterogeneousList, DessiaObject
import dessia_common.workflow as wf

# Standard cars homogeneous dataset from the Internet
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(tests.Car.from_csv(csv_cars))
# When attribute _features is specified in class CarWithFeatures
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
all_cars_with_features = HeterogeneousList(tests.CarWithFeatures.from_csv(csv_cars))

# Auto-generated heterogeneous small dataset with nb_clusters clusters of points in nb_dims dimensions
mean_borns = (-50, 50)
std_borns = (-2, 2)
small_clustesters_heterogeneous = HeterogeneousList(
    tests.ClusTester_d5.create_dataset(nb_clusters = 10, nb_points = 250, mean_borns = mean_borns, std_borns = std_borns) +
    tests.ClusTester_d4.create_dataset(nb_clusters = 10, nb_points = 250, mean_borns = mean_borns, std_borns = std_borns) +
    tests.ClusTester_d3.create_dataset(nb_clusters = 10, nb_points = 250, mean_borns = mean_borns, std_borns = std_borns))

# Auto-generated heterogeneous large dataset with nb_clusters clusters of points in nb_dims dimensions
mean_borns = (-50, 50)
std_borns = (-2, 2)
big_clustesters_heterogeneous = HeterogeneousList(
    tests.ClusTester_d9.create_dataset(nb_clusters = 10, nb_points = 500, mean_borns = mean_borns, std_borns = std_borns) +
    tests.ClusTester_d7.create_dataset(nb_clusters = 10, nb_points = 500, mean_borns = mean_borns, std_borns = std_borns) +
    tests.ClusTester_d8.create_dataset(nb_clusters = 10, nb_points = 500, mean_borns = mean_borns, std_borns = std_borns))

# Build CategorizedLists
clustered_cars_without = cluster.CategorizedList.from_dbscan(all_cars_without_features, eps=40)
clustered_cars_with = cluster.CategorizedList.from_dbscan(all_cars_with_features, eps=40)
aggclustest_clustered = cluster.CategorizedList.from_agglomerative_clustering(big_clustesters_heterogeneous, n_clusters=10)
kmeanstest_clustered = cluster.CategorizedList.from_kmeans(small_clustesters_heterogeneous, n_clusters=10, scaling=False)

# Test ClusterResults instances on platform
# clustered_cars_without._check_platform()
# clustered_cars_with._check_platform()
# aggclustest_clustered._check_platform()
# kmeanstest_clustered._check_platform()

# Test plots outside platform
# clustered_cars_without.plot()
# clustered_cars_with.plot()
# aggclustest_clustered.plot()
# kmeanstest_clustered.plot()


# =============================================================================
# JSON TESTS
# =============================================================================
dict_cars_without = clustered_cars_without.to_dict(use_pointers=True)
dict_cars_with = clustered_cars_with.to_dict(use_pointers=True)
dict_aggclustest = aggclustest_clustered.to_dict(use_pointers=True)
dict_kmeanstest = kmeanstest_clustered.to_dict(use_pointers=True)

# Cars without features
json_dict = json.dumps(dict_cars_without)
decoded_json = json.loads(json_dict)
deserialized_object = clustered_cars_without.dict_to_object(decoded_json)

# Cars with features
json_dict = json.dumps(dict_cars_with)
decoded_json = json.loads(json_dict)
deserialized_object = clustered_cars_with.dict_to_object(decoded_json)

# Small dataset
json_dict = json.dumps(dict_aggclustest)
decoded_json = json.loads(json_dict)
deserialized_object = aggclustest_clustered.dict_to_object(decoded_json)

# Large dataset
json_dict = json.dumps(dict_kmeanstest)
decoded_json = json.loads(json_dict)
deserialized_object = kmeanstest_clustered.dict_to_object(decoded_json)


# =============================================================================
# TESTS IN WORKFLOWS: CARS WITHOUT FEATURES
# =============================================================================
data_method = wf.MethodType(class_=tests.Car, name='from_csv')
block_data = wf.ClassMethod(method_type=data_method, name='data load')

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

categorized_list_method = wf.MethodType(class_=cluster.CategorizedList, name='from_dbscan')
block_cluster = wf.ClassMethod(method_type=categorized_list_method, name='labelling elements of list')

block_workflow = [block_data, block_heterogeneous_list, block_cluster]
pipe_worflow = [wf.Pipe(block_data.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_cluster.inputs[0])]
workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

workflow_run = workflow.run({workflow.index(block_data.inputs[0]): pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv'),
                              workflow.index(block_cluster.inputs[1]): 40})

# workflow._check_platform()
# workflow.plot()
# workflow.display_settings()
# workflow_run.output_value.plot()

# JSON TESTS
dict_workflow = workflow.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow)
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)

# dict_workflow_run = workflow_run.to_dict(use_pointers=True)
# json_dict = json.dumps(dict_workflow_run)
# decoded_json = json.loads(json_dict)
# deserialized_object = workflow_run.dict_to_object(decoded_json)


# =============================================================================
# TESTS IN WORKFLOWS: CARS WITH FEATURES
# =============================================================================
data_method = wf.MethodType(class_=tests.CarWithFeatures, name='from_csv')
block_data = wf.ClassMethod(method_type=data_method, name='data load')

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

categorized_list_method = wf.MethodType(class_=cluster.CategorizedList, name='from_dbscan')
block_cluster = wf.ClassMethod(method_type=categorized_list_method, name='labelling elements of list')

block_workflow = [block_data, block_heterogeneous_list, block_cluster]
pipe_worflow = [wf.Pipe(block_data.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_cluster.inputs[0])]
workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

workflow_run = workflow.run({workflow.index(block_data.inputs[0]): pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv'),
                              workflow.index(block_cluster.inputs[1]): 40})


# workflow._check_platform()
# workflow.plot()
# workflow.display_settings()
# workflow_run.output_value.plot()

# JSON TESTS
dict_workflow = workflow.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow)
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)

# dict_workflow_run = workflow_run.to_dict(use_pointers=True)
# json_dict = json.dumps(dict_workflow_run)
# decoded_json = json.loads(json_dict)
# deserialized_object = workflow_run.dict_to_object(decoded_json)


# =============================================================================
# TESTS IN WORKFLOWS: CLUSTESTERS SMALL DATASET
# =============================================================================
data_method = wf.MethodType(class_=tests.ClusTester_d5, name='create_dataset')
block_data_d5 = wf.ClassMethod(method_type=data_method, name='data d5')

data_method = wf.MethodType(class_=tests.ClusTester_d4, name='create_dataset')
block_data_d4 = wf.ClassMethod(method_type=data_method, name='data d4')

data_method = wf.MethodType(class_=tests.ClusTester_d3, name='create_dataset')
block_data_d3 = wf.ClassMethod(method_type=data_method, name='data d3')

block_concatenate = wf.Concatenate(3)

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

categorized_list_method = wf.MethodType(class_=cluster.CategorizedList, name='from_dbscan')
block_cluster = wf.ClassMethod(method_type=categorized_list_method, name='labelling elements of list')

block_workflow = [block_data_d5, block_data_d4, block_data_d3, block_concatenate, block_heterogeneous_list, block_cluster]

pipe_worflow = [wf.Pipe(block_data_d5.outputs[0], block_concatenate.inputs[0]),
                wf.Pipe(block_data_d4.outputs[0], block_concatenate.inputs[1]),
                wf.Pipe(block_data_d3.outputs[0], block_concatenate.inputs[2]),
                wf.Pipe(block_concatenate.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_cluster.inputs[0])]

workflow = wf.Workflow(block_workflow, pipe_worflow, block_cluster.outputs[0])

workflow_run = workflow.run({workflow.index(block_data_d5.inputs[0]): 10, workflow.index(block_data_d5.inputs[1]): 500,
                             workflow.index(block_data_d5.inputs[2]): mean_borns, workflow.index(block_data_d5.inputs[3]): std_borns,

                             workflow.index(block_data_d4.inputs[0]): 10, workflow.index(block_data_d4.inputs[1]): 500,
                             workflow.index(block_data_d4.inputs[2]): mean_borns, workflow.index(block_data_d4.inputs[3]): std_borns,

                             workflow.index(block_data_d3.inputs[0]): 10, workflow.index(block_data_d3.inputs[1]): 500,
                             workflow.index(block_data_d3.inputs[2]): mean_borns, workflow.index(block_data_d3.inputs[3]): std_borns,

                             workflow.index(block_cluster.inputs[1]): 40})



# workflow._check_platform()
# workflow.plot()
# workflow.display_settings()
# workflow_run.output_value.plot()

# JSON TESTS
dict_workflow = workflow.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow)
decoded_json = json.loads(json_dict)
deserialized_object = workflow.dict_to_object(decoded_json)

dict_workflow_run = workflow_run.to_dict(use_pointers=True)
json_dict = json.dumps(dict_workflow_run)
decoded_json = json.loads(json_dict)
deserialized_object = workflow_run.dict_to_object(decoded_json)

# gg = workflow_run._display_from_selector('plot_data')
# json.dumps(workflow_run.to_dict())
# workflow_run._displays
