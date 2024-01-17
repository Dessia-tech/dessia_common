from dessia_common.forms import DEF_SO, StandaloneObject
from dessia_common.schemas.core import MethodSchema
import unittest
from parameterized import parameterized


class TestMethodSchemas(unittest.TestCase):
    @parameterized.expand([
        ("add_standalone_object", 1),
        ("count_until", 2),
        ("add_float", 1),
        ("generate_from_text", 1),
        ("generate_from_bin", 1),
        ("method_without_arg", 0),
        ("ill_defined_method", 1)
    ])
    def test_number_of_arguments(self, method_name, expected_number):
        method = getattr(StandaloneObject, method_name)
        schema = MethodSchema(method)
        self.assertEqual(len(schema.property_schemas), expected_number)

    @parameterized.expand([
        ("add_standalone_object", True),
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

    @parameterized.expand([
        ("add_standalone_object", 4),
        ("count_until", 4),
        ("add_float", 3),
        ("generate_from_text", 4),
        ("generate_from_bin", 4),
        ("method_without_arg", 2),
        ("ill_defined_method", 6)
    ])
    def test_check_list(self, method_name, expected_number):
        method = getattr(StandaloneObject, method_name)
        schema = MethodSchema(method)
        self.assertEqual(len(schema.check_list()), expected_number)


class TestComputedSchemas(unittest.TestCase):
    def setUp(self) -> None:
        self.schemas = DEF_SO.method_schemas

    def test_schemas_statics(self):
        self.assertEqual(len(self.schemas), 7)
        self.assertSetEqual(set(DEF_SO._allowed_methods), set(self.schemas.keys()))

    @parameterized.expand([
        ("add_standalone_object", ["0"]),
        ("count_until", ["0"]),
        ("add_float", []),
        ("generate_from_text", ["0"]),
        ("generate_from_bin", ["0"]),
        ("method_without_arg", []),
        ("ill_defined_method", ["0"])
    ])
    def test_required_args(self, method_name, expected_requirements):
        computed_schema = self.schemas[method_name]
        self.assertListEqual(computed_schema["required"], expected_requirements)

    def test_count_until_method(self):
        computed_schemas = self.schemas["count_until"]
        duration = computed_schemas["properties"]["0"]
        raise_error = computed_schemas["properties"]["1"]
        self.assertEqual(computed_schemas["description"], "Test long execution with a user-input duration.")
        self.assertDictEqual(duration, {"title": "Duration", "editable": True, "python_typing": "float",
                                        "description": "Duration of the method in s", "type": "number"})
        self.assertDictEqual(raise_error, {
            "title": "Raise Error", "editable": True, "type": "boolean", "default_value": False,
            "description": "Whether the computation should raise an error or not at the end",
            "python_typing": "bool"
        })

    @parameterized.expand([
        ("add_standalone_object", 1),
        ("count_until", 2),
        ("add_float", 1),
        ("generate_from_text", 1),
        ("generate_from_bin", 1),
        ("method_without_arg", 0),
        ("ill_defined_method", 4)
    ])
    def test_number_of_properties(self, method_name, expected_number):
        computed_schema = self.schemas[method_name]
        self.assertEqual(len(computed_schema["properties"]), expected_number)


if __name__ == '__main__':
    unittest.main(verbosity=2)
