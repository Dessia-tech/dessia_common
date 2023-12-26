from dessia_common.schemas.core import ClassProperty, MethodTypeProperty, AttributeTypeProperty, SchemaAttribute
from dessia_common.forms import StandaloneObject
from dessia_common.typings import MethodType, ClassMethodType, AttributeType, ClassAttributeType
from typing import Type

import unittest
from parameterized import parameterized


CUSTOM_CLASS = SchemaAttribute(name="custom_class")
ATTRIBUTE = SchemaAttribute(name="attribute")
METHOD = SchemaAttribute(name="method")


class TestStructures(unittest.TestCase):
    @parameterized.expand([
        (ClassProperty(annotation=Type, attribute=CUSTOM_CLASS), 'Type'),
        (
            ClassProperty(annotation=Type[StandaloneObject], attribute=CUSTOM_CLASS),
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
            AttributeTypeProperty(annotation=AttributeType[StandaloneObject], attribute=ATTRIBUTE),
            "attributes", "AttributeType[dessia_common.forms.StandaloneObject]",
            "Type[dessia_common.forms.StandaloneObject]"
        ),
        (
            AttributeTypeProperty(annotation=ClassAttributeType[StandaloneObject], attribute=ATTRIBUTE),
            "class_attributes", "ClassAttributeType[dessia_common.forms.StandaloneObject]",
            "Type[dessia_common.forms.StandaloneObject]"
        ),
        (
            MethodTypeProperty(annotation=MethodType[StandaloneObject], attribute=METHOD),
            "methods", "MethodType[dessia_common.forms.StandaloneObject]",
            "Type[dessia_common.forms.StandaloneObject]"
        ),
        (
            MethodTypeProperty(annotation=ClassMethodType[StandaloneObject], attribute=METHOD),
            "class_methods", "ClassMethodType[dessia_common.forms.StandaloneObject]",
            "Type[dessia_common.forms.StandaloneObject]"
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
