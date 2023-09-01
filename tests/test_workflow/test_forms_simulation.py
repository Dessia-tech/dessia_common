from scripts.workflow.forms_simulation import workflow_run, workflow_
import unittest
from parameterized import parameterized


class TestWorkflowFeatures(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @parameterized.expand([
        (workflow_,),
        (workflow_run,)
    ])
    def test_hashes(self, object_):
        serialized_object = object_.to_dict()
        deserialized_object = object_.dict_to_object(dict_=serialized_object)
        self.assertEqual(hash(object_), hash(deserialized_object))

    @parameterized.expand([
        ('(0, 0, 0)', ['0']),
        ('(0, 0, 1)', ['0']),
        ('(0, 0, 2)', ['(1, 1, 0)']),
        ('(0, 0, 3)', ['1']),
        ('(0, 1, 0)', ['(1, 0, 0)']),
        ('(1, 0, 0)', ['(0, 1, 0)']),
        ('(1, 1, 0)', ['(0, 0, 2)', '(4, 0, 0)']),
        ('(1, 1, 1)', []),
        ('(3, 0, 1)', ['1']),
        ('(3, 0, 2)', ['0']),
        ('(4, 0, 0)', []),
        ('(4, 1, 0)', []),
        ('0', ['(0, 0, 0)', '(0, 0, 1)', '(3, 0, 2)']),
        ('1', ['(0, 0, 3)', '(3, 0, 1)']),
    ])
    def test_variables_match(self, variable, expected_match):
        variables_match = workflow_.match_variables()
        self.assertIn(variable, variables_match)
        self.assertEqual(variables_match[variable], expected_match)

    @parameterized.expand([
        ("#/values/1", "Test"),
        ("#/values/0", 100),
        ("#/values/1/0", None)
    ])
    def test_getattr(self, path, expected_result):
        if expected_result is not None:
            self.assertEqual(workflow_run._get_from_path(path), expected_result)
        else:
            self.assertRaises(AttributeError, workflow_run._get_from_path(path))

    def test_check_platform(self):
        workflow_._check_platform()
        workflow_run._check_platform()

        copied_workflow_run = workflow_run.copy()
        copied_workflow_run._check_platform()
