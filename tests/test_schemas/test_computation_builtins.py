from dessia_common.schemas.core import BuiltinProperty, MeasureProperty
from dessia_common.measures import Distance

import unittest
from parameterized import parameterized


class TestBuiltins(unittest.TestCase):
    @parameterized.expand([
        (BuiltinProperty(annotation=float, attribute="floating"), False, "number", "float"),
        (BuiltinProperty(annotation=int, attribute="integer"), True, "number", "int"),
        (BuiltinProperty(annotation=str, attribute="string"), False, "string", "str"),
        (BuiltinProperty(annotation=bool, attribute="boolean"), False, "boolean", "bool"),
    ])
    def test_computation(self, schema, editable, expected_type, expected_typing):
        computed_schema = schema.to_dict(title="Title", editable=editable, description="Desc")
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["title"], "Title")
        self.assertEqual(computed_schema["editable"], editable)
        self.assertEqual(computed_schema["description"], "Desc")

    @parameterized.expand([
        (MeasureProperty(annotation=Distance, attribute="distance"), "number", "dessia_common.measures.Distance", "m")
    ])
    def test_measures(self, schema, expected_type, expected_python_typing, expected_units):
        computed_schema = schema.to_dict(title="Measure", editable=True, description="Testing Measures")
        self.assertEqual(computed_schema["type"], expected_type)
        self.assertEqual(computed_schema["python_typing"], expected_python_typing)
        self.assertEqual(computed_schema["title"], "Measure")
        self.assertEqual(computed_schema["editable"], True)
        self.assertEqual(computed_schema["description"], "Testing Measures")
        self.assertEqual(computed_schema["si_unit"], expected_units)


if __name__ == '__main__':
    unittest.main(verbosity=2)
