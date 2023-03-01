
from dessia_common.forms import EmbeddedSubobject, StandaloneObject, EnhancedEmbeddedSubobject, DEF_SO
from dessia_common.utils.helpers import full_classname

subobject_classname = full_classname(object_=EmbeddedSubobject, compute_for='class')
enhanced_classname = full_classname(object_=EnhancedEmbeddedSubobject, compute_for='class')

assert subobject_classname == 'dessia_common.forms.EmbeddedSubobject'
assert enhanced_classname == 'dessia_common.forms.EnhancedEmbeddedSubobject'
serialized_union = 'Union[{}, {}]'.format(subobject_classname, enhanced_classname)

# Test deep_attr
assert DEF_SO._get_from_path("#/standalone_subobject/floatarg") == 0.3
assert DEF_SO._get_from_path("#/embedded_subobject/embedded_list/2") == 3
assert DEF_SO._get_from_path("#/object_list/0/floatarg") == 0.3
assert DEF_SO._get_from_path("#/standalone_subobject/name") == "EmbeddedSubobject1"
assert DEF_SO._get_from_path("name") == "Standalone Object Demo"

# Test to_dict/dict_to_object
d = DEF_SO.to_dict()
obj = StandaloneObject.dict_to_object(d)
DEF_SO._check_platform()

# Testing not empty cad displays
assert DEF_SO._display_from_selector('cad').data

assert DEF_SO == obj

# Test serialization
d = DEF_SO.to_dict(use_pointers=True)
assert '$ref' in d['union_arg'][1]
o = StandaloneObject.dict_to_object(d)
assert not isinstance(o.union_arg[1], dict)

obj.to_xlsx('test')
print("script unit_tests.py passed")
