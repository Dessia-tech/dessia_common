"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import itertools
import random
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_large
from dessia_common.core import HeterogeneousList, DessiaFilter, FiltersList

# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(all_cars_no_feat)
# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
RandData_heterogeneous = HeterogeneousList(rand_data_large)

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
all_cars_with_features.plot()
all_cars_without_features.plot()
RandData_heterogeneous.plot()

# Check platform for datasets
all_cars_with_features._check_platform()
all_cars_without_features._check_platform()
RandData_heterogeneous._check_platform()

# Check for __getitem__ and __str__
print(all_cars_with_features)

assert(all_cars_with_features[2] == all_cars_with_features.dessia_objects[2])
assert(all_cars_with_features[:10] == HeterogeneousList(all_cars_wi_feat[:10]))

try:
    all_cars_without_features[[True, False, True]]
    raise ValueError("boolean indexes of len 3 should not be able to index HeterogeneousLists of len 406")
except Exception as e:
    assert(e.args[0] == "Cannot index HeterogeneousList object of len 406 with a list of boolean of len 3")

assert(all_cars_without_features[:3][[True, False, True]] ==
       HeterogeneousList([all_cars_without_features[0], all_cars_without_features[2]]))

# Filters creation
weight_val = 2000.
mpg_big_val = 100.
mpg_low_val = 40.
filter_1 = DessiaFilter('weight', 'le', weight_val)
filter_2 = DessiaFilter('mpg', 'ge', mpg_big_val)
filter_3 = DessiaFilter('mpg', 'ge', mpg_low_val)
filters_list = [filter_1, filter_3]
print(FiltersList(filters_list, logical_operator="or"))

# Or testing
filters_list_fun = lambda x: ((getattr(value, 'weight') <= weight_val or
                               getattr(value, 'mpg') >= mpg_big_val or
                               getattr(value, 'mpg') >= mpg_low_val)
                              for value in all_cars_no_feat)

assert(all(item in all_cars_without_features.filtering(filters_list, logical_operator="or")
           for item in list(itertools.compress(all_cars_no_feat, filters_list_fun(all_cars_no_feat)))))

# And with non empty result
filters_list = [filter_1, filter_3]
filters_list_fun = lambda x: ((getattr(value, 'weight') <= weight_val and getattr(value, 'mpg') >= mpg_low_val)
                              for value in all_cars_no_feat)
assert(all(item in all_cars_without_features.filtering(filters_list, logical_operator="and")
           for item in list(itertools.compress(all_cars_no_feat, filters_list_fun(all_cars_no_feat)))))

# And with empty result
filters_list = [filter_1, filter_2]
assert(all_cars_without_features.filtering(filters_list, "and") == HeterogeneousList())

# Xor
filter_1 = DessiaFilter('weight', 'le', weight_val)
filter_3 = DessiaFilter('mpg', 'ge', mpg_low_val)
filters_list = [filter_1, filter_2, filter_3]
filters_list_fun = lambda x: ((getattr(value, 'weight') <= weight_val
                               and not getattr(value, 'mpg') >= mpg_big_val
                               and not getattr(value, 'mpg') >= mpg_low_val) or
                              (not getattr(value, 'weight') <= weight_val
                               and getattr(value, 'mpg') >= mpg_big_val
                               and not getattr(value, 'mpg') >= mpg_low_val) or
                              (not getattr(value, 'weight') <= weight_val
                               and not getattr(value, 'mpg') >= mpg_big_val
                               and getattr(value, 'mpg') >= mpg_low_val)
                          for value in all_cars_no_feat)
assert(all(item in all_cars_without_features.filtering(filters_list, logical_operator="xor")
           for item in list(itertools.compress(all_cars_no_feat, filters_list_fun(all_cars_no_feat)))))

try:
    all_cars_without_features.filtering(filters_list, logical_operator="blurps")
    raise ValueError("'blurps' should not work for logical_operator attribute in FiltersList")
except Exception as e:
    assert(e.args[0] == "'blurps' str for 'logical_operator' attribute is not a use case")


# Tests for empty HeterogeneousList
empty_list = HeterogeneousList()
print(empty_list)
assert(empty_list[0] == [])
assert(empty_list[:] == [])
assert(empty_list[[False, True]] == [])
assert(empty_list + empty_list == HeterogeneousList())
assert(empty_list + all_cars_without_features == all_cars_without_features)
assert(all_cars_without_features + empty_list == all_cars_without_features)
assert(len(empty_list) == 0)
assert(empty_list.matrix == [])
assert(empty_list.common_attributes == [])
empty_list.sort(0)
assert(empty_list == HeterogeneousList())
empty_list.sort("weight")
assert(empty_list == HeterogeneousList())

try:
    empty_list.plot_data()
    raise ValueError("plot_data should not work on empty HeterogeneousLists")
except Exception as e:
    assert(e.__class__.__name__ == "LinAlgError")
try:
    empty_list.singular_values()
    raise ValueError("singular_values should not work on empty HeterogeneousLists")
except Exception as e:
    assert(e.__class__.__name__ == "LinAlgError")

# Tests sort
all_cars_with_features.sort('weight', ascend=False)
assert(all_cars_with_features[0].weight == max(all_cars_with_features.get_attribute_values('weight')))

all_cars_without_features.sort(2)
assert(all_cars_without_features.common_attributes[2] == "displacement")
assert(all_cars_without_features[0].displacement == min(all_cars_without_features.get_column_values(2)))
