from dessia_common.schemas.core import BuiltinProperty, MeasureProperty, SchemaAttribute
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


class TestBuiltins(unittest.TestCase):
    @parameterized.expand([
        (BuiltinProperty(annotation=float, attribute=FLOAT_ATTRIBUTE), "number", "float"),
        (BuiltinProperty(annotation=int, attribute=INT_ATTRIBUTE), "number", "int"),
        (BuiltinProperty(annotation=str, attribute=STR_ATTRIBUTE), "string", "str"),
        (BuiltinProperty(annotation=bool, attribute=BOOL_ATTRIBUTE), "boolean", "bool"),
    ])
    def test_computation(self, schema, expected_type, expected_typing):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["python_typing"], expected_typing)
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
        self.assertEqual(computed_schema["python_typing"], expected_python_typing)
        self.assertEqual(computed_schema["title"], schema.attribute.title)
        self.assertEqual(computed_schema["editable"], schema.attribute.editable)
        self.assertEqual(computed_schema["description"], schema.attribute.documentation["desc"])
        self.assertEqual(computed_schema["si_unit"], expected_units)


if __name__ == '__main__':
    unittest.main(verbosity=2)
