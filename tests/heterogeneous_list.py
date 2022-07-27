"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import time
import random
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_large
from dessia_common.core import HeterogeneousList, DessiaFilter

# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(all_cars_no_feat)
# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
RandData_heterogeneous = HeterogeneousList(rand_data_large)

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

idx = random.randint(0, len(RandData_heterogeneous) - 3)
all_cars_without_features.extend(RandData_heterogeneous[idx:idx+2])
assert(all(item in all_cars_without_features
            for item in HeterogeneousList(all_cars_without_features+RandData_heterogeneous[idx:idx+2])))
all_cars_without_features = HeterogeneousList(all_cars_no_feat)

# Tests for plot_data
# all_cars_with_features.plot()
# all_cars_without_features.plot()
# RandData_heterogeneous.plot()

# Check platform for datasets
# all_cars_with_features._check_platform()
# all_cars_without_features._check_platform()
# RandData_heterogeneous._check_platform()

# Check for __getitem__, __str__ and sorts
print(all_cars_with_features)
print(all_cars_with_features[2])
all_cars_with_features.sort('weight', ascend=False)
print(all_cars_with_features[:10])
all_cars_without_features.sort(0)
print(f"sort name : {all_cars_without_features.common_attributes[0]}")
print(all_cars_without_features[:10])
try:
    all_cars_without_features[[True, False, True]]
except Exception as e:
    print(e)
print(all_cars_without_features[:3][[True, False, True]])

# Tests for Filters
print("Filters tests")

# Filters creation
new_filter_1 = DessiaFilter('weight', 'le', 2000)
new_filter_2 = DessiaFilter('mpg', 'ge', 100.)
new_filter_3 = DessiaFilter('mpg', 'ge', 30.)
filters_list = [new_filter_1, new_filter_2, new_filter_3]

# Or testing
print(all_cars_without_features.filtering([new_filter_1]))
heavy_list = HeterogeneousList(all_cars_without_features.dessia_objects*100)
t = time.time()
for _ in range(1):
    heavy_list = heavy_list.filtering(filters_list*20, "or")
print("OR", f"{len(filters_list*20)} filters computed on HetererogeneousList of length {len(heavy_list)} " +
      f"in {time.time()-t} sec")
print(heavy_list)

# And with non empty result
filters_list = [new_filter_1, new_filter_3]
print("AND NON EMPTY", all_cars_without_features.filtering(filters_list, "and"))

# And with empty result
filters_list = [new_filter_1, new_filter_2]
print("AND EMPTY", all_cars_without_features.filtering(filters_list, "and"))

# Xor
new_filter_1 = DessiaFilter('weight', 'le', 1700)
new_filter_3 = DessiaFilter('mpg', 'ge', 40.)
filters_list = [new_filter_1, new_filter_2, new_filter_3]
print("XOR", all_cars_without_features.filtering(filters_list, "xor"))

# get_columns and get_attr
print(all_cars_without_features.get_column_values(0)[:15])
print(all_cars_without_features.get_attribute_values("weight")[:15])

# Tests for empty HeterogeneousList
print(HeterogeneousList())
empty_list = HeterogeneousList()
empty_list[0]
empty_list[:]
empty_list[[False, True]]
empty_list + empty_list
empty_list + all_cars_without_features
all_cars_without_features + empty_list
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

