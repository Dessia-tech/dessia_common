from dessia_common.schemas.core import ClassSchema
from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues

import unittest
from parameterized import parameterized


class TestStandaloneObject(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = ClassSchema(StandaloneObject)

    @parameterized.expand([
        ("standalone_subobject", None),
        ("embedded_subobject", None),
        ("dynamic_dict", None),
        ("float_dict", None),
        ("string_dict", None),
        ("tuple_arg", (None, None)),
        ("object_list", None),
        ("subobject_list", None),
        ("builtin_list", None),
        ("union_arg", None),
        ("subclass_arg", None),
        ("array_arg", None),
        ("name", "Standalone Object Demo")
    ])
    def test_default_values(self, attribute, expected_default):
        self.assertEqual(self.schema.property_schemas[attribute].default_value(), expected_default)


class TestStandaloneObjectWithDefaultValues(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = ClassSchema(StandaloneObjectWithDefaultValues)

    @parameterized.expand([
        ("dynamic_dict", None),
        ("float_dict", None),
        ("string_dict", None),
        ("tuple_arg", ("Default Tuple", 0)),
        ("object_list", None),
        ("subobject_list", None),
        ("builtin_list", None),
        ("union_arg", None),
        ("array_arg", None)
    ])
    def test_default_values(self, attribute, expected_default):
        self.assertEqual(self.schema.property_schemas[attribute].default_value(), expected_default)

    @parameterized.expand([
        ("standalone_subobject", {
            "name": "EmbeddedSubobject1", "object_class": "dessia_common.forms.StandaloneBuiltinsSubobject",
            "floatarg": 0.3, "distarg": {"object_class": "dessia_common.measures.Distance", "value": 0.51}
        }),
        ("embedded_subobject", {
            "name": "Embedded Subobject10", "object_class": "dessia_common.forms.EmbeddedSubobject",
            "embedded_list": [0, 1, 2, 3, 4]
        }),
        ("subclass_arg", {
            "name": "Inheriting Standalone Subobject1",
            "object_class": "dessia_common.forms.InheritingStandaloneSubobject",
            "floatarg": 0.1, "distarg": {"object_class": "dessia_common.measures.Distance", "value": 0.7}
        })
    ])
    def test_complex_default_values(self, attribute, expected_partial_dict):
        default_value = self.schema.property_schemas[attribute].default_value()
        for key, value in expected_partial_dict.items():
            self.assertEqual(default_value[key], value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
