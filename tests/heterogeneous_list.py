"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import random
import pkg_resources
from dessia_common import tests
from dessia_common.core import HeterogeneousList, DessiaFilter, FiltersList

# Standard cars homogeneous dataset from the Internet
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(tests.Car.from_csv(csv_cars))
# When attribute _features is specified in class CarWithFeatures
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
all_cars_with_features = HeterogeneousList(tests.CarWithFeatures.from_csv(csv_cars))

# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
RandData_heterogeneous = HeterogeneousList(tests.RandDataD9.create_dataset(nb_clusters=10, nb_points=500) +
                                              tests.RandDataD7.create_dataset(nb_clusters=10, nb_points=500) +
                                              tests.RandDataD8.create_dataset(nb_clusters=10, nb_points=500))

# Test on auto-generated attributes
car_matrix_with = all_cars_with_features.matrix
car_matrix_without = all_cars_without_features.matrix
heter_matrix = RandData_heterogeneous.matrix

# Test on matrices
idx = random.randint(0, len(all_cars_with_features) - 1)
assert(all(item in all_cars_with_features.matrix[idx]
           for item in [getattr(all_cars_with_features.dessia_objects[idx], attr)
                        for attr in all_cars_with_features.common_attributes]))

idx = random.randint(0, len(all_cars_without_features) - 1)
assert(all(item in all_cars_without_features.matrix[idx]
           for item in [getattr(all_cars_without_features.dessia_objects[idx], attr)
                        for attr in all_cars_without_features.common_attributes]))

idx = random.randint(0, len(RandData_heterogeneous) - 1)
assert(all(item in RandData_heterogeneous.matrix[idx]
           for item in [getattr(RandData_heterogeneous.dessia_objects[idx], attr)
                        for attr in RandData_heterogeneous.common_attributes]))

# Tests for plot_data
# all_cars_with_features.plot()
# all_cars_without_features.plot()
# RandData_heterogeneous.plot()

# Check platform for datasets
all_cars_with_features._check_platform()
all_cars_without_features._check_platform()
RandData_heterogeneous._check_platform()

# Check for sorts
print(all_cars_with_features)
print(all_cars_with_features[0:10])
print(all_cars_with_features[2])
print(all_cars_with_features[:10])
all_cars_with_features.sort('weight', ascend=False)
print(all_cars_with_features[:10])
all_cars_without_features.sort(0)
print(f"sort name : {all_cars_without_features.common_attributes[0]}")
print(all_cars_without_features[:10])

# Filters
print("Filters tests")
try:
    all_cars_without_features[[True, False, True]]
except Exception as e:
    print(e)

print(all_cars_without_features[:3][[True, False, True]])

# Filters creation
new_filter_1 = DessiaFilter('weight', 'le', 2000)
new_filter_2 = DessiaFilter('mpg', 'ge', 100.)
new_filter_3 = DessiaFilter('mpg', 'ge', 30.)
filters_list = FiltersList([new_filter_1, new_filter_2, new_filter_3], "or")
print(filters_list)

# Or testing
print(new_filter_1.apply(all_cars_without_features))
print(new_filter_2.apply(all_cars_without_features))
gg=filters_list.apply(all_cars_without_features)
print("or", gg[::int(len(gg)/30)])

# And with non empty result
filters_list = FiltersList([new_filter_1, new_filter_3], "and")
gg=filters_list.apply(all_cars_without_features)
print("and non empty", gg)

# And with empty result
filters_list = FiltersList([new_filter_1, new_filter_2], "and")
gg=filters_list.apply(all_cars_without_features)
print("and empty", gg)

# Xor
new_filter_1 = DessiaFilter('weight', 'le', 1700)
new_filter_3 = DessiaFilter('mpg', 'ge', 40.)
filters_list = FiltersList([new_filter_1, new_filter_2, new_filter_3], "xor")
print(filters_list)
gg=filters_list.apply(all_cars_without_features)
print("xor", gg)

print(gg.get_column_values(3))
print(gg.get_attribute_values("displacement"))

# Tests for empty HeterogeneousList
print(HeterogeneousList())
empty_list = HeterogeneousList()
empty_list[0]
empty_list[:]
empty_list[[False, True]]
empty_list + empty_list
empty_list + gg
gg + empty_list
len(empty_list)
empty_list.matrix
empty_list.common_attributes
try:
    empty_list.plot_data()
except Exception as e:
    print(e)
try:
    empty_list.singular_values()
except Exception as e:
    print(e)
empty_list.sort(0)
empty_list.sort("weight")



# =============================================================================
# CARS WITH FEATURES
# =============================================================================
import json
from dessia_common import tests
import dessia_common.workflow as wf

data_method = wf.MethodType(class_=tests.CarWithFeatures, name='from_csv')
block_data = wf.ClassMethod(method_type=data_method, name='data load')

block_heterogeneous_list = wf.InstantiateModel(model_class=HeterogeneousList, name='heterogeneous list of data')

filters_list = [DessiaFilter('weight', "<=", 2000), DessiaFilter('mpg', ">=", 30)]

filter_method = wf.MethodType(class_=FiltersList, name='from_filters_list')
block_list_filters = wf.ClassMethod(method_type=filter_method, name='create filters list')

apply_method = wf.MethodType(class_=FiltersList, name='apply')
block_apply = wf.ModelMethod(method_type=apply_method, name='apply filter')

block_workflow = [block_data, block_heterogeneous_list, block_list_filters, block_apply]
pipe_worflow = [wf.Pipe(block_data.outputs[0], block_heterogeneous_list.inputs[0]),
                wf.Pipe(block_list_filters.outputs[0], block_apply.inputs[0]),
                wf.Pipe(block_heterogeneous_list.outputs[0], block_apply.inputs[1]),
                ]
workflow = wf.Workflow(block_workflow, pipe_worflow, block_apply.outputs[0])

workflow_run = workflow.run({
    workflow.index(block_data.inputs[0]): pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv'),
    workflow.index(block_list_filters.inputs[0]): filters_list,
    workflow.index(block_list_filters.inputs[1]): "or"})

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

# print(all_cars_with_features[ all_cars_with_features.tolist(attr) == 70 ])


