from dessia_common.utils.schemas import schema_chunk, ClassSchema
import unittest
from typing import List
from dessia_common import DessiaObject
from dessia_common.models.forms import standalone_object

try:
    schema_chunk(List, "Bare List", True, "Testing Bare List")
except NotImplementedError:
    pass

value = {
    "type": "array",
    "python_typing": "List[__builtins__.int]",
    "items": {
        "type": "number",
        "python_typing": "builtins.int",
        "title": "One arg List",
        "editable": True,
        "description": "",
    },
    "title": "One arg List",
    "editable": True,
    "description": "Testing one arg List",
}
assert schema_chunk(List[int], "One arg List", True, "Testing one arg List") == value

try:
    schema_chunk(List[int, str], "Several arg List", True, "Testing several arg List")
except TypeError:
    pass

class_schema = ClassSchema(DessiaObject)
schema = class_schema.write()
assert len(schema) == 7
assert len(schema["properties"]) == 1 and "name" in schema["properties"]


# --- Schema computation ---
schema = {
    "definitions": {},
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "standalone_subobject",
        "embedded_subobject",
        "dynamic_dict",
        "float_dict",
        "string_dict",
        "tuple_arg",
        "intarg",
        "strarg",
        "object_list",
        "subobject_list",
        "builtin_list",
        "union_arg",
        "subclass_arg",
        "array_arg",
    ],
    "properties": {
        "standalone_subobject": {
            "title": "Standalone Subobject",
            "editable": True,
            "python_typing": "dessia_common.forms.StandaloneSubobject",
            "type": "object",
            "standalone_in_db": True,
            "classes": ["dessia_common.forms.StandaloneSubobject"],
            "description": "A dev subobject that is standalone_in_db",
        },
        "embedded_subobject": {
            "title": "Embedded Subobject",
            "editable": True,
            "python_typing": "dessia_common.forms.EmbeddedSubobject",
            "type": "object",
            "standalone_in_db": False,
            "classes": ["dessia_common.forms.EmbeddedSubobject"],
            "description": "A dev subobject that isn't standalone_in_db",
        },
        "dynamic_dict": {
            "title": "Dynamic Dict",
            "editable": True,
            "python_typing": "Dict[__builtins__.str, __builtins__.bool]",
            "type": "object",
            "patternProperties": {".*": {"type": "boolean"}},
            "description": "A variable length dict",
        },
        "float_dict": {
            "title": "Float Dict",
            "editable": True,
            "python_typing": "Dict[__builtins__.str, __builtins__.float]",
            "type": "object",
            "description": "",
            "patternProperties": {".*": {"type": "number"}},
        },
        "string_dict": {
            "title": "String Dict",
            "editable": True,
            "python_typing": "Dict[__builtins__.str, __builtins__.str]",
            "type": "object",
            "description": "",
            "patternProperties": {".*": {"type": "string"}},
        },
        "tuple_arg": {
            "title": "Tuple Arg",
            "editable": True,
            "python_typing": "Tuple[__builtins__.str, __builtins__.int]",
            "additionalItems": False,
            "type": "array",
            "items": [{"type": "string"}, {"type": "number"}],
            "description": "A heterogeneous sequence",
        },
        "intarg": {
            "title": "Intarg",
            "editable": True,
            "python_typing": "builtins.int",
            "description": "",
            "type": "number",
        },
        "strarg": {
            "title": "Strarg",
            "editable": True,
            "python_typing": "builtins.str",
            "description": "",
            "type": "string",
        },
        "object_list": {
            "title": "Object List",
            "editable": True,
            "python_typing": "List[dessia_common.forms.StandaloneSubobject]",
            "description": "",
            "type": "array",
            "items": {
                "title": "Object List",
                "editable": True,
                "python_typing": "dessia_common.forms.StandaloneSubobject",
                "description": "",
                "type": "object",
                "standalone_in_db": True,
                "classes": ["dessia_common.forms.StandaloneSubobject"],
            },
        },
        "subobject_list": {
            "title": "Subobject List",
            "editable": True,
            "python_typing": "List[dessia_common.forms.EmbeddedSubobject]",
            "description": "",
            "type": "array",
            "items": {
                "title": "Subobject List",
                "editable": True,
                "python_typing": "dessia_common.forms.EmbeddedSubobject",
                "description": "",
                "type": "object",
                "standalone_in_db": False,
                "classes": ["dessia_common.forms.EmbeddedSubobject"],
            },
        },
        "builtin_list": {
            "title": "Builtin List",
            "editable": True,
            "python_typing": "List[__builtins__.int]",
            "description": "",
            "type": "array",
            "items": {
                "title": "Builtin List",
                "editable": True,
                "description": "",
                "python_typing": "builtins.int",
                "type": "number",
            },
        },
        "union_arg": {
            "title": "Union Arg",
            "editable": True,
            "python_typing": "List[Union[dessia_common.forms.EmbeddedSubobject,"
                             " dessia_common.forms.EnhancedEmbeddedSubobject]]",
            "type": "array",
            "description": "",
            "items": {
                "title": "Union Arg",
                "editable": True,
                "description": "",
                "python_typing": "Union[dessia_common.forms.EmbeddedSubobject,"
                                 " dessia_common.forms.EnhancedEmbeddedSubobject]",
                "type": "object",
                "classes": [
                    "dessia_common.forms.EmbeddedSubobject",
                    "dessia_common.forms.EnhancedEmbeddedSubobject",
                ],
                "standalone_in_db": False,
            },
        },
        "subclass_arg": {
            "title": "Subclass Arg",
            "editable": True,
            "python_typing": "InstanceOf[dessia_common.forms.StandaloneSubobject]",
            "type": "object",
            "description": "",
            "instance_of": "dessia_common.forms.StandaloneSubobject",
            "standalone_in_db": True,
        },
        "array_arg": {
            "title": "Array Arg",
            "editable": True,
            "python_typing": "List[List[__builtins__.float]]",
            "description": "",
            "type": "array",
            "items": {
                "type": "array",
                "python_typing": "List[__builtins__.float]",
                "items": {
                    "title": "Array Arg",
                    "editable": True,
                    "description": "",
                    "python_typing": "builtins.float",
                    "type": "number",
                },
            },
        },
        "name": {
            "title": "Name",
            "editable": True,
            "description": "",
            "python_typing": "builtins.str",
            "type": "string",
            "default_value": "Standalone Object Demo",
        },
    },
    "standalone_in_db": True,
    "description": "Dev Object for testing purpose",
    "python_typing": "<class 'dessia_common.forms.StandaloneObject'>",
    "classes": ["dessia_common.forms.StandaloneObject"],
    "whitelist_attributes": [],
}

computed_schema = standalone_object.schema()
try:
    assert computed_schema == schema
except AssertionError as err:
    for key, value in computed_schema["properties"].items():
        if value != schema["properties"][key]:
            print("\n==", key, "property failing ==\n")
            for subkey, subvalue in value.items():
                if subkey in schema["properties"][key]:
                    check_value = schema["properties"][key][subkey]
                    if subvalue != check_value:
                        print("Problematic key :", {subkey})
                        print(
                            "Computed value : ",
                            subvalue,
                            "\nCheck value : ",
                            check_value,
                        )
            print("\n", value)
            print("\n", schema["properties"][key])
            raise err


print("schemas.py test has passed")


# class TestListSchemas(unittest.TestCase):
#     def test_bare_list(self):
#         self.assertRaises(schema_chunk(List, "Bare List", True, "Testing Bare List"), TypeError)
#
#     def test_one_arg_list(self):
#         value = {'type': 'array', 'python_typing': 'List[__builtins__.int]',
#                  'items': {'type': 'number',
#                            'python_typing': 'builtins.int',
#                            'title': 'One arg List',
#                            'editable': True,
#                            'description': ''},
#                  'title': 'One arg List',
#                  'editable': True,
#                  'description': 'Testing one arg List'}
#         self.assertDictEqual(schema_chunk(List[int], "One arg List", True, "Testing one arg List"), value)
#
#     def test_several_arg_list(self):
#         self.assertRaises(schema_chunk(List[int, str], "Several arg List", True, "Testing several arg List"), TypeError)
#
#
# if __name__ == '__main__':
#     unittest.main()
