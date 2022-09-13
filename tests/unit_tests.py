from dessia_common.forms import EmbeddedSubobject, StandaloneObject, EnhancedEmbeddedSubobject
from dessia_common import full_classname, DessiaObject
from dessia_common.models.forms import standalone_object

subobject_classname = full_classname(object_=EmbeddedSubobject, compute_for='class')
enhanced_classname = full_classname(object_=EnhancedEmbeddedSubobject, compute_for='class')

assert subobject_classname == 'dessia_common.forms.EmbeddedSubobject'
assert enhanced_classname == 'dessia_common.forms.EnhancedEmbeddedSubobject'
serialized_union = 'Union[{}, {}]'.format(subobject_classname, enhanced_classname)

# Test deep_attr
assert standalone_object._get_from_path("#/standalone_subobject/floatarg") == 3.78
assert standalone_object._get_from_path("#/embedded_subobject/embedded_list/2") == 3
assert standalone_object._get_from_path("#/object_list/0/floatarg") == 666.999
assert standalone_object._get_from_path("intarg") == 5

# Test to_dict/dict_to_object
d = standalone_object.to_dict()
obj = StandaloneObject.dict_to_object(d)
standalone_object._check_platform()

# Testing not empty cad displays
assert standalone_object._display_from_selector('cad').data

assert standalone_object == obj

# Test serialization
d = standalone_object.to_dict(use_pointers=True)
assert '$ref' in d['subobject_list'][0]
o = DessiaObject.dict_to_object(d)
assert not isinstance(o.subobject_list[0], dict)

obj.to_xlsx('test')
print("script unit_tests.py passed")
