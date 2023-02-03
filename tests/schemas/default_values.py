from dessia_common.schemas.core import ClassSchema
from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues

schema = ClassSchema(StandaloneObject)

assert schema.property_schemas["standalone_subobject"].default_value() is None
assert schema.property_schemas["embedded_subobject"].default_value() is None
assert schema.property_schemas["dynamic_dict"].default_value() is None
assert schema.property_schemas["float_dict"].default_value() is None
assert schema.property_schemas["string_dict"].default_value() is None
assert schema.property_schemas["tuple_arg"].default_value() == (None, None)
assert schema.property_schemas["object_list"].default_value() is None
assert schema.property_schemas["subobject_list"].default_value() is None
assert schema.property_schemas["builtin_list"].default_value() is None
assert schema.property_schemas["union_arg"].default_value() is None
assert schema.property_schemas["subclass_arg"].default_value() is None
assert schema.property_schemas["array_arg"].default_value() is None
assert schema.property_schemas["name"].default_value() == "Standalone Object Demo"

schema = ClassSchema(StandaloneObjectWithDefaultValues)

subobject_default_value = schema.property_schemas["standalone_subobject"].default_value()
assert subobject_default_value["name"] == "EmbeddedSubobject1"
assert subobject_default_value["object_class"] == "dessia_common.forms.StandaloneBuiltinsSubobject"
assert subobject_default_value["floatarg"] == 0.3
assert subobject_default_value["distarg"]["value"] == 0.51

subobject_default_value = schema.property_schemas["embedded_subobject"].default_value()
assert subobject_default_value["name"] == "Embedded Subobject10"
assert subobject_default_value["object_class"] == "dessia_common.forms.EmbeddedSubobject"
assert subobject_default_value["embedded_list"] == [0, 1, 2, 3, 4]

assert schema.property_schemas["dynamic_dict"].default_value() is None
assert schema.property_schemas["float_dict"].default_value() is None
assert schema.property_schemas["string_dict"].default_value() is None
assert schema.property_schemas["tuple_arg"].default_value() == ("Default Tuple", 0)
assert schema.property_schemas["object_list"].default_value() is None
assert schema.property_schemas["subobject_list"].default_value() is None
assert schema.property_schemas["builtin_list"].default_value() is None
assert schema.property_schemas["union_arg"].default_value() is None

subobject_default_value = schema.property_schemas["subclass_arg"].default_value()
assert subobject_default_value["name"] == "Inheriting Standalone Subobject1"
assert subobject_default_value["object_class"] == "dessia_common.forms.InheritingStandaloneSubobject"
assert subobject_default_value["strarg"] == "-1"
assert subobject_default_value["floatarg"] == 0.1
assert subobject_default_value["distarg"]["value"] == 0.7

assert schema.property_schemas["array_arg"].default_value() is None
assert schema.property_schemas["name"].default_value() == "Standalone Object Demo"


print("script 'default_values.py' has passed.")
