import unittest
from parameterized import parameterized
from dessia_common.models.workflows.forms_workflow import workflow_


class TestRunSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = workflow_.method_schemas["run"]

    @parameterized.expand([
        ("required", ["0", "3", "5"]),
        ("method", True),
        ("type", "object"),
        ("order", ['0', '1', '2', '3', '4', '5'])
    ])
    def test_items(self, key, value):
        self.assertEqual(self.schema[key], value)

    @parameterized.expand([
        ("title", "ForEach - Binding: Workflow Block - optimization Value"),
        ("python_typing", "int"),
        ("editable", True),
        ("type", "number"),
        ("default_value", 3),
        ("description", "value that will be added to model's intarg attribute")
    ])
    def test_properties(self, key, value):
        properties = self.schema["properties"]
        self.assertEqual(len(properties), 6)
        self.assertEqual(properties["2"][key], value)


class TestStartSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = workflow_.method_schemas["start_run"]

    @parameterized.expand([
        ("required", []),
    ])
    def test_items(self, key, value):
        self.assertEqual(self.schema[key], value)

    @parameterized.expand([
        ("title", "Shared Name"),
        ("python_typing", "str"),
        ("editable", True),
        ("type", "string"),
        ("default_value", "Shared Name"),
        ("description", "")
    ])
    def test_properties(self, key, value):
        properties = self.schema["properties"]
        self.assertEqual(len(properties), 6)
        self.assertEqual(properties["4"][key], value)


class TestWorkflowSchemasEqualities(unittest.TestCase):
    def test_equalities(self):
        run_schema = workflow_.method_schemas["run"]
        start_schema = workflow_.method_schemas["start_run"]
        del run_schema["required"]
        del start_schema["required"]
        self.assertEqual(run_schema, start_schema)
