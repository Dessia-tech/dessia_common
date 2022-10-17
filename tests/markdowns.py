"""
Tests for markdowns
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.forms import StandaloneObjectWithDefaultValues
from dessia_common.datatools import HeterogeneousList, CategorizedList

standalone_object = StandaloneObjectWithDefaultValues()
standalone_object.to_markdown()

dataset = HeterogeneousList(all_cars_no_feat)
dataset.to_markdown()

clustered_dataset = CategorizedList(all_cars_no_feat)
clustered_dataset.to_markdown()
