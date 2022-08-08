"""
Tests for dessia_common.utils.helpers
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.core import HeterogeneousList
from dessia_common.utils.helpers import concatenate

values_list = [all_cars_no_feat, all_cars_no_feat]
values_hlist = [HeterogeneousList(all_cars_no_feat), HeterogeneousList(all_cars_no_feat)]
values_dict = [{'0': all_cars_no_feat, '1': all_cars_no_feat}, {'2': all_cars_no_feat, '3': all_cars_no_feat}]

assert(concatenate(values_list) == all_cars_no_feat + all_cars_no_feat)
assert(concatenate(values_hlist) == HeterogeneousList(all_cars_no_feat + all_cars_no_feat))
assert(concatenate(values_dict) == {'0': all_cars_no_feat, '1': all_cars_no_feat,
                                    '2': all_cars_no_feat, '3': all_cars_no_feat})
