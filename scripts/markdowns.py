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
cars_md = dataset.to_markdown()

clustered_dataset = CategorizedList(all_cars_no_feat)
clustered_dataset.to_markdown()

funky_md = MarkdownWriter().matrix_table([[all_cars_no_feat[0:2], 2, all_cars_no_feat[0]],
                                         [all_cars_no_feat[:5], 3, all_cars_no_feat[10]]],
                                         ['subobject_list', 'object_list', 'carname'])


ref_funky_md = ("| Subobject_list | Object_list | Carname |\n| ------ | ------ | ------ |\n| 2 Cars | 2 | " +
                "Chevrolet Chevelle Malibu |\n| 5 Cars | 3 | Citroen DS-21 Pallas |\n")

assert(all(assertion in cars_md for assertion in ['Model', "0.119", "2625.0"]))
assert(funky_md == ref_funky_md)


writer = MarkdownWriter()
headings = ['Introduction', 'Methodology', 'Results', 'Conclusion']
md = writer.table_of_contents(headings)
for heading in headings:
    assert(heading in md)

print("script 'markdowns.py' has passed")
