"""
Tests for dessia_common.utils.helpers.
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.utils.helpers import concatenate

values_list = [all_cars_no_feat, all_cars_no_feat]
values_hlist = [Dataset(all_cars_no_feat), Dataset(all_cars_no_feat)]
heterogeneous_list = [all_cars_no_feat, Dataset(all_cars_no_feat)]
int_vector = [1, 2, 3, 4, 5, 6]

assert(concatenate(values_list) == all_cars_no_feat + all_cars_no_feat)
assert(concatenate(values_hlist) == Dataset(all_cars_no_feat + all_cars_no_feat))

try:
    concatenate(heterogeneous_list)
except Exception as e:
    assert(e.args[0] == "Block Concatenate only defined for operands of the same type.")

try:
    concatenate(int_vector)
except Exception as e:
    assert(e.args[0] == ("Block Concatenate only defined for classes with 'extend' method"))
