import unittest
from parameterized import parameterized
from dessia_common.models.workflows.forms_workflow import workflow_


class TestRunSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = workflow_.method_schemas["run"]
        self.properties = self.schema["spareProperties"]

    def test_properties(self):
        self.assertEqual(len(self.properties), 5)

    @parameterized.expand([
        ("title", "ForEach - Binding: Workflow Block - optimization Value"),
        ("pythonTyping", "int"),
        ("editable", True),
        ("type", "number"),
        ("defaultValue", 3),
        ("description", "value that will be added to model's intarg attribute"),
        ("key", "2")
    ])
    def test_property(self, key, value):
        self.assertEqual(self.properties[2][key], value)
