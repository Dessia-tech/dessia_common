from dessia_common.forms import DEF_SO, StandaloneObject
from dessia_common.schemas.core import MethodSchema
import unittest
from parameterized import parameterized

schemas = DEF_SO.method_schemas

assert len(schemas) == 8
assert set(DEF_SO._allowed_methods) == set(schemas.keys())


class TestMethodSchemas(unittest.TestCase):
    @parameterized.expand([
        ("add_standalone_object", 1, True),
        ("add_embedded_object", 1, True),
        ("count_until", 2, True),
        ("add_float", 1, True),
        ("generate_from_text", 1, True),
        ("generate_from_bin", 1, True),
        ("method_without_arg", 0, True),
        ("ill_defined_method", 4, False)
    ])
    def test_checks(self, method_name, number_of_arguments, is_truthy):
        method = getattr(StandaloneObject, method_name)
        schema = MethodSchema(method)
        self.assertEqual(len(schema.check_list()), number_of_arguments)
        self.assertEqual(len(schema.property_schemas), number_of_arguments)
        if is_truthy:
            self.assertFalse(schema.check_list().checks_above_level("error"))
        else:
            self.assertTrue(schema.check_list().checks_above_level("error"))

computed_count_until_schema = schemas["count_until"]
assert computed_count_until_schema["type"] == "object"
assert computed_count_until_schema["required"] == ["0"]
assert computed_count_until_schema["description"] == "Test long execution with a customizable duration."
assert len(computed_count_until_schema["properties"]) == 2
assert computed_count_until_schema["properties"]["0"] == {"title": "Duration", "editable": True, "python_typing": "float",
                                                 "description": "Duration of the method in s", "type": "number"}
assert computed_count_until_schema["properties"]["1"] == {
    "title": "Raise Error", "editable": True, "type": "boolean", "default_value": False,
    "description": "Wether the computation should raise an error or not at the end", "python_typing": "bool"
}

print("script 'methods.py' has passed.")

