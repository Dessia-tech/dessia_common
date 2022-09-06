from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues
import dessia_common.utils.jsonschema as jss

# --- Default values ---
jsonschema = StandaloneObject.jsonschema()

assert jss.chose_default(jsonschema["properties"]["standalone_subobject"]) is None
assert jss.chose_default(jsonschema["properties"]["embedded_subobject"]) is None
assert jss.chose_default(jsonschema["properties"]["dynamic_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["float_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["string_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["tuple_arg"]) == [None, None]
assert jss.chose_default(jsonschema["properties"]["intarg"]) is None
assert jss.chose_default(jsonschema["properties"]["strarg"]) is None
assert jss.chose_default(jsonschema["properties"]["object_list"]) is None
assert jss.chose_default(jsonschema["properties"]["subobject_list"]) is None
assert jss.chose_default(jsonschema["properties"]["builtin_list"]) is None
assert jss.chose_default(jsonschema["properties"]["union_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["subclass_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["array_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["name"]) is None

jsonschema = StandaloneObjectWithDefaultValues.jsonschema()

subobject_jsonschema = jsonschema["properties"]["standalone_subobject"]
assert jss.chose_default(subobject_jsonschema["name"]) == "StandaloneSubobject1"
assert jss.chose_default(subobject_jsonschema["object_class"]) == "dessia_common.forms.StandaloneSubobject"
assert jss.chose_default(subobject_jsonschema["floatarg"]) == 1.7

subobject_jsonschema = jsonschema["properties"]["embedded_subobject"]
assert jss.chose_default(subobject_jsonschema["name"]) == "Embedded Subobject10"
assert jss.chose_default(subobject_jsonschema["object_class"]) == "dessia_common.forms.EmbeddedSubobject"
assert jss.chose_default(subobject_jsonschema["embedded_list"]) == [0, 1, 2, 3, 4]

assert jss.chose_default(jsonschema["properties"]["dynamic_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["float_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["string_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["tuple_arg"]) == ("Default Tuple", 0)
assert jss.chose_default(jsonschema["properties"]["intarg"]) is None  # TODO Is it ?
assert jss.chose_default(jsonschema["properties"]["strarg"]) is None  # TODO Is it ?
assert jss.chose_default(jsonschema["properties"]["object_list"]) is None
assert jss.chose_default(jsonschema["properties"]["subobject_list"]) is None
assert jss.chose_default(jsonschema["properties"]["builtin_list"]) is None
assert jss.chose_default(jsonschema["properties"]["union_arg"]) is None

subobject_jsonschema = jsonschema["properties"]["subclass_arg"]
assert jss.chose_default(subobject_jsonschema["name"]) == "Inheriting Standalone Subobject1"
assert jss.chose_default(subobject_jsonschema["object_class"]) == "dessia_common.forms.InheritingStandaloneSubobject"
assert jss.chose_default(subobject_jsonschema["strarg"]) == "-1"
assert jss.chose_default(subobject_jsonschema["floatarg"]) == 0.7

assert jss.chose_default(jsonschema["properties"]["array_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["name"]) is None  # TODO Is it ?


# --- Datatypes

jsonschema = StandaloneObject.jsonschema()

assert jss.datatype_from_jsonschema(jsonschema["properties"]["standalone_subobject"]) == "standalone_object"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["embedded_subobject"]) == "embedded_object"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["dynamic_dict"]) == "dynamic_dict"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["float_dict"]) == "dynamic_dict"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["string_dict"]) == "dynamic_dict"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["tuple_arg"]) == "heterogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["intarg"]) == "builtin"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["strarg"]) == "builtin"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["object_list"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["subobject_list"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["builtin_list"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["union_arg"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["subclass_arg"]) == "instance_of"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["array_arg"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["name"]) == "builtin"

print("test script jsonschema.py has passed")



