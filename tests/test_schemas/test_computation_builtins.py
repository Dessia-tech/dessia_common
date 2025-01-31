from typing import Literal
from dessia_common.schemas.core import BuiltinProperty, MeasureProperty, SchemaAttribute, EnumProperty
from dessia_common.typings import KeyOf
from dessia_common.measures import Distance

import unittest
from parameterized import parameterized

DUMMY_DOCUMENTATION = {"desc": "Desc", "type": "", "annotation": ""}
FLOAT_ATTRIBUTE = SchemaAttribute(name="floating", editable=False, title="Title", documentation=DUMMY_DOCUMENTATION)
INT_ATTRIBUTE = SchemaAttribute(name="integer", editable=True, title="Title", documentation=DUMMY_DOCUMENTATION)
STR_ATTRIBUTE = SchemaAttribute(name="string", editable=False, title="Title", documentation=DUMMY_DOCUMENTATION)
BOOL_ATTRIBUTE = SchemaAttribute(name="boolean", editable=False, title="Title", documentation=DUMMY_DOCUMENTATION)

MEASURE_DOCUMENTATION = {"desc": "Testing Measures", "type": "", "annotation": ""}
MEASURE_ATTRIBUTE = SchemaAttribute(name="distance", editable=True, title="Measure", documentation=DUMMY_DOCUMENTATION)

KEYOF_DUMMY = {"red": 1, "green": 2, "blue": 3}
KEYOF_ATTRIBUTE = SchemaAttribute(name="keyof", title="Key Of")
LITERAL_ATTRIBUTE = SchemaAttribute(name="literal", title="Literal")


class TestBuiltins(unittest.TestCase):
    @parameterized.expand([
        (BuiltinProperty(annotation=float, attribute=FLOAT_ATTRIBUTE), "number", "float"),
        (BuiltinProperty(annotation=int, attribute=INT_ATTRIBUTE), "number", "int"),
        (BuiltinProperty(annotation=str, attribute=STR_ATTRIBUTE), "string", "str"),
        (BuiltinProperty(annotation=bool, attribute=BOOL_ATTRIBUTE), "boolean", "bool")
    ])
    def test_computation(self, schema, expected_type, expected_typing):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["pythonTyping"], expected_typing)
        self.assertEqual(computed_schema["title"], schema.attribute.title)
        self.assertEqual(computed_schema["editable"], schema.attribute.editable)
        self.assertEqual(computed_schema["description"], schema.attribute.documentation["desc"])

    @parameterized.expand([
        (MeasureProperty(annotation=Distance, attribute=MEASURE_ATTRIBUTE),
         "number", "dessia_common.measures.Distance", "m")
    ])
    def test_measures(self, schema, expected_type, expected_python_typing, expected_units):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["pythonTyping"], expected_python_typing)
        self.assertEqual(computed_schema["title"], schema.attribute.title)
        self.assertEqual(computed_schema["editable"], schema.attribute.editable)
        self.assertEqual(computed_schema["description"], schema.attribute.documentation["desc"])
        self.assertEqual(computed_schema["si_unit"], expected_units)

    @parameterized.expand([
        (EnumProperty(annotation=KeyOf[KEYOF_DUMMY], attribute=KEYOF_ATTRIBUTE),
         "string",
         "Literal['red', 'green', 'blue']",
         ("red", "green", "blue")),
        (EnumProperty(annotation=Literal["a", "b"], attribute=LITERAL_ATTRIBUTE),
         "string",
         "Literal['a', 'b']",
         ("a", "b"))
    ])
    def test_enums(self, schema, expected_type, expected_typing, expected_values):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["pythonTyping"], expected_typing)
        self.assertEqual(computed_schema["allowedValues"], expected_values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
