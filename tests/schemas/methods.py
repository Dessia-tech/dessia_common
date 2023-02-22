from dessia_common.forms import DEF_SO, StandaloneObject
from dessia_common.schemas.core import MethodSchema
import unittest
from parameterized import parameterized


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
    def test_number_of_arguments(self, method_name, expected_number_of_arguments, is_truthy):
        method = getattr(StandaloneObject, method_name)
        schema = MethodSchema(method)
        self.assertEqual(len(schema.check_list()), expected_number_of_arguments)
        self.assertEqual(len(schema.property_schemas), expected_number_of_arguments)
        if is_truthy:
            self.assertFalse(schema.check_list().checks_above_level("error"))
        else:
            self.assertTrue(schema.check_list().checks_above_level("error"))

    @parameterized.expand([
        ("add_standalone_object", True),
        ("add_embedded_object", True),
        ("count_until", True),
        ("add_float", True),
        ("generate_from_text", True),
        ("generate_from_bin", True),
        ("method_without_arg", True),
        ("ill_defined_method", False)
    ])
    def test_truthiness(self, method_name, should_be_truthy):
        method = getattr(StandaloneObject, method_name)
        schema = MethodSchema(method)
        if should_be_truthy:
            self.assertFalse(schema.check_list().checks_above_level("error"))
        else:
            self.assertTrue(schema.check_list().checks_above_level("error"))


# class TestComputedSchemas(unittest.TestCase):
#     def setUp(self) -> None:
#         self.schemas = DEF_SO.method_schemas
#
#     def test_schemas_statics(self):
#         assert len(self.schemas) == 8
#         assert set(DEF_SO._allowed_methods) == set(self.schemas.keys())
#
#     @parameterized.expand([
#         ("add_standalone_object", True),
#         ("add_embedded_object", True),
#         ("count_until", True),
#         ("add_float", True),
#         ("generate_from_text", True),
#         ("generate_from_bin", True),
#         ("method_without_arg", True),
#         ("ill_defined_method", False)
#     ])
#     def test_required_args(self, method_name, expected_number):
#         computed_schema = self.schemas[method_name]
#         method = getattr(StandaloneObject, method_name)
#         schema = MethodSchema(method)
#         if should_be_truthy:
#             self.assertFalse(schema.check_list().checks_above_level("error"))
#         else:
#             self.assertTrue(schema.check_list().checks_above_level("error"))
#
# assert computed_count_until_schema["type"] == "object"
# assert computed_count_until_schema["required"] == ["0"]
# assert computed_count_until_schema["description"] == "Test long execution with a user-input duration."
# assert len(computed_count_until_schema["properties"]) == 2
# assert computed_count_until_schema["properties"]["0"] == {"title": "Duration", "editable": True, "python_typing": "float",
#                                                  "description": "Duration of the method in s", "type": "number"}
# assert computed_count_until_schema["properties"]["1"] == {
#     "title": "Raise Error", "editable": True, "type": "boolean", "default_value": False,
#     "description": "Wether the computation should raise an error or not at the end", "python_typing": "bool"
# }

print("script 'methods.py' has passed.")

