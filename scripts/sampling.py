"""
sampling.py package testing.

"""

from typing import Type
from dessia_common.serialization import deserialize_argument
from dessia_common.tests import RandDataD6
from dessia_common.datatools.sampling import ClassSampler
from dessia_common.optimization import FixedAttributeValue, BoundedAttributeValue

sampled_attributes = [BoundedAttributeValue('p_3', 0.1, 0.5, 5),
                      BoundedAttributeValue('p_5', 1500, 5000, 2),
                      BoundedAttributeValue('p_4', 4, 8, 4),
                      BoundedAttributeValue('p_1', 25, 50)]

constant_attributes = [FixedAttributeValue('name', 'sampled_randata'),
                       FixedAttributeValue('p_2', -2.1235),
                       FixedAttributeValue('p_6', 31.1111111)]

randata_sampling = ClassSampler(sampled_class=RandDataD6,
                                sampled_attributes=sampled_attributes,
                                constant_attributes=constant_attributes)

randata_sampling.make_doe('fullfact')
randata_sampling.make_doe(20, 'lhs')
randata_sampling.make_doe(samples=2000, method='montecarlo')
try:
    test_hlist = randata_sampling.make_doe(samples=2000, method='truc')
except NotImplementedError as e:
    assert (e.args[0] == ("Method 'truc' is not implemented in "
                          "<class 'dessia_common.datatools.sampling.ClassSampler'>._get_doe method."))

randata_sampling._check_platform()

assert deserialize_argument(Type, "dessia_common.tests.RandDataD6") == RandDataD6

print("script 'clustering.py' has passed")
