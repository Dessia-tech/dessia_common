"""
Tests for dessia_common.utils.helpers
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.utils.helpers import concatenate

values_list = [all_cars_no_feat, all_cars_no_feat]
values_hlist = [Dataset(all_cars_no_feat), Dataset(all_cars_no_feat)]
multi_types = [all_cars_no_feat, Dataset(all_cars_no_feat)]
wrong_type = [1,2,3,4,5,6]

assert(concatenate(values_list) == all_cars_no_feat + all_cars_no_feat)
assert(concatenate(values_hlist) == Dataset(all_cars_no_feat + all_cars_no_feat))

try:
    concatenate(multi_types)
except Exception as e:
    assert(e.args[0] == "Block Concatenate only defined for operands of the same type.")

try:
    concatenate(wrong_type)
except Exception as e:
    assert(e.args[0] == ("Block Concatenate only defined for classes with 'extend' method"))
