from dessia_common.schemas.core import ClassProperty, MethodTypeProperty, AttributeTypeProperty
from dessia_common.forms import StandaloneObject
from dessia_common.typings import MethodType, ClassMethodType, AttributeType, ClassAttributeType
from typing import Type

import unittest
from parameterized import parameterized


class TestStructures(unittest.TestCase):
    @parameterized.expand([
        (ClassProperty(annotation=Type, attribute="custom_class"), 'Type'),
        (
            ClassProperty(annotation=Type[StandaloneObject], attribute="custom_class"),
            'Type[dessia_common.forms.StandaloneObject]'
        )
    ])
    def test_classes(self, schema, expected_typing):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], "object")
        self.assertEqual(computed_schema["is_class"], True)
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["properties"], {'name': {'type': 'string'}})

    @parameterized.expand([
        (
            AttributeTypeProperty(annotation=AttributeType[StandaloneObject], attribute="attribute"),
            "attributes", "AttributeType[dessia_common.forms.StandaloneObject]",
            "dessia_common.forms.StandaloneObject"
        ),
        (
            AttributeTypeProperty(annotation=ClassAttributeType[StandaloneObject], attribute="attribute"),
            "class_attributes", "ClassAttributeType[dessia_common.forms.StandaloneObject]",
            "dessia_common.forms.StandaloneObject"
        ),
        (
            MethodTypeProperty(annotation=MethodType[StandaloneObject], attribute="method"),
            "methods", "MethodType[dessia_common.forms.StandaloneObject]",
            "dessia_common.forms.StandaloneObject"
        ),
        (
            MethodTypeProperty(annotation=ClassMethodType[StandaloneObject], attribute="method"),
            "class_methods", "ClassMethodType[dessia_common.forms.StandaloneObject]",
            "dessia_common.forms.StandaloneObject"
        )
    ])
    def test_attributes(self, schema, expected_type, expected_typing, expected_class):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], "object")
        self.assertEqual(computed_schema["is_attribute"], True)
        self.assertEqual(computed_schema["attribute_type"], expected_type)
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["properties"]["class_"]["type"], "object")
        self.assertEqual(computed_schema["properties"]["class_"]["python_typing"], expected_class)


if __name__ == '__main__':
    unittest.main(verbosity=2)
