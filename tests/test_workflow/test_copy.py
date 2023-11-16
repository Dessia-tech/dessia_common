import unittest
from parameterized import parameterized
from dessia_common.models.workflows.forms_workflow import workflow_


class TestWorkflowCopy(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow_copy = workflow_.copy()

    def test_check_platform(self):
        workflow_.check_platform()
        self.workflow_copy.check_platform()

    @parameterized.expand([
        "pipes",
        "nonblock_variables"
    ])
    def test_attributes_len(self, name):
        value = getattr(workflow_, name)
        copied_value = getattr(self.workflow_copy, name)
        self.assertEqual(len(value), len(copied_value))

    def variable_names(self):
        variable_names_are_equal = [p.input_variable.name == cp.input_variable.name
                                    for p, cp in zip(workflow_.pipes, self.workflow_copy.pipes)]
        self.assertTrue(all(variable_names_are_equal))


