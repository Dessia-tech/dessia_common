from dessia_common.core import DessiaObject
from dessia_common.schemas.core import ClassProperty, MethodTypeProperty, AttributeTypeProperty, SchemaAttribute, \
    CustomClass
from dessia_common.forms import StandaloneObject
from dessia_common.typings import MethodType, ClassMethodType, AttributeType, ClassAttributeType
from typing import Type

import unittest
from parameterized import parameterized


CUSTOM_CLASS = SchemaAttribute(name="custom_class")
CUSTOM_CLASS_DEFAULT = SchemaAttribute(name="custom_class", editable=True, title="CustomClass", default_value=None)
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
        self.assertEqual(computed_schema["pythonTyping"], expected_typing)
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
        self.assertEqual(computed_schema["attributeType"], expected_type)
        self.assertEqual(computed_schema["pythonTyping"], expected_typing)
        self.assertEqual(computed_schema["properties"]["class_"]["type"], "object")
        self.assertEqual(computed_schema["properties"]["class_"]["pythonTyping"], expected_class)

    @parameterized.expand([
        (CustomClass(annotation=DessiaObject, attribute=CUSTOM_CLASS_DEFAULT),
         "object", "dessia_common.core.DessiaObject"),
        (CustomClass(annotation=DessiaObject, attribute=CUSTOM_CLASS),
         "object", "dessia_common.core.DessiaObject")
    ])
    def test_custom_classes(self, schema, expected_type, expected_python_typing):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["pythonTyping"], expected_python_typing)
        self.assertEqual(computed_schema["title"], schema.attribute.title)
        self.assertEqual(computed_schema["editable"], schema.attribute.editable)


if __name__ == '__main__':
    unittest.main(verbosity=2)
