"""
Tests for markdowns
"""
from dessia_common.models import all_cars_no_feat
from dessia_common.forms import StandaloneObjectWithDefaultValues
from dessia_common.datatools import HeterogeneousList, CategorizedList
from dessia_common.exports import MarkdownWriter

standalone_object = StandaloneObjectWithDefaultValues()
print(standalone_object.to_markdown())

dataset = HeterogeneousList(all_cars_no_feat)
print(dataset.to_markdown())

clustered_dataset = CategorizedList(all_cars_no_feat)
clustered_dataset.to_markdown()

print(MarkdownWriter(StandaloneObjectWithDefaultValues()).matrix_table([[all_cars_no_feat[0:2],2, all_cars_no_feat[0]],
                                                                        [all_cars_no_feat[:5],3, all_cars_no_feat[10]]],
                                                                      ['subobject_list','object_list', 'carname']))
