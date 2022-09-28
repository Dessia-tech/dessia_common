"""
sampling.py package testing.

"""

import json
from dessia_common.datatools import HeterogeneousList
from dessia_common.tests import RandDataD6
from dessia_common.sampling import Sampler
from dessia_common.optimization import FixedAttributeValue, BoundedAttributeValue


sampled_attributes = [BoundedAttributeValue('p_3', 0.1, 0.5, 5),
                      BoundedAttributeValue('p_5', 1500, 5000, 2),
                      BoundedAttributeValue('p_4', 4, 8, 4),
                      BoundedAttributeValue('p_1', 25, 50)]

constant_attributes = [FixedAttributeValue('name', 'sampled_randata'),
                       FixedAttributeValue('p_2', -2.1235),
                       FixedAttributeValue('p_6', 31.1111111)]

randata_sampling = Sampler(sampled_class=RandDataD6,
                           sampled_attributes=sampled_attributes,
                           constant_attributes=constant_attributes)

# test_hlist = randata_sampling.make_doe('fullfact')
# test_hlist = randata_sampling.make_doe('lhs', samples = 20)
# test_hlist.plot()

json_dict = json.dumps(randata_sampling.to_dict())
decoded_json = json.loads(json_dict)
deserialized_object = randata_sampling.dict_to_object(decoded_json)
randata_sampling._check_platform()
