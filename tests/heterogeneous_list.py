"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import json
import random
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_middl
from dessia_common.datatools import HeterogeneousList

# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(all_cars_no_feat)
# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
RandData_heterogeneous = HeterogeneousList(rand_data_middl)

# Compute one common_attributes
all_cars_without_features.common_attributes

# Check platform for datasets
all_cars_with_features._check_platform()
all_cars_without_features._check_platform()
RandData_heterogeneous._check_platform()

# Test __getitem__
picked_list = (all_cars_with_features[250:] +
               RandData_heterogeneous[:50][[1, 4, 6, 10, 25]][[True, False, True, True, False]])
assert(picked_list._common_attributes is None)
assert(picked_list._matrix is None)
assert(picked_list[-1] == rand_data_middl[10])
assert(RandData_heterogeneous.common_attributes == ['p_1', 'p_2', 'p_3', 'p_4', 'test_prop'])
try:
    all_cars_without_features[[True, False, True]]
    raise ValueError("boolean indexes of len 3 should not be able to index HeterogeneousLists of len 406")
except Exception as e:
    assert(e.args[0] == "Cannot index HeterogeneousList object of len 406 with a list of boolean of len 3")

# Test on matrice
idx = random.randint(0, len(all_cars_without_features) - 1)
assert(all(item in all_cars_without_features.matrix[idx]
            for item in [getattr(all_cars_without_features.dessia_objects[idx], attr)
                        for attr in all_cars_without_features.common_attributes]))

# Tests for displays
hlist_cars_plot_data = all_cars_without_features.plot_data()
assert(json.dumps(hlist_cars_plot_data[0].to_dict())[150:200] == 'acceleration": 12.0, "model": 70.0}, {"mpg": 15.0,')
assert(json.dumps(hlist_cars_plot_data[1].to_dict())[10500:10548] == 'celeration": 12.5, "model": 72.0}, {"mpg": 13.0,')
assert(json.dumps(hlist_cars_plot_data[2].to_dict())[50:100] == 'te_names": ["Index of reduced basis vector", "Sing')
print(all_cars_with_features)

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
    assert(e.__class__.__name__ == "ValueError")
try:
    empty_list.singular_values()
    raise ValueError("singular_values should not work on empty HeterogeneousLists")
except Exception as e:
    assert(e.__class__.__name__ == "ValueError")

# Tests sort
all_cars_with_features.sort('weight', ascend=False)
assert(all_cars_with_features[0].weight == max(all_cars_with_features.get_attribute_values('weight')))

all_cars_without_features.sort(2)
assert(all_cars_without_features.common_attributes[2] == "displacement")
assert(all_cars_without_features[0].displacement == min(all_cars_without_features.get_column_values(2)))

