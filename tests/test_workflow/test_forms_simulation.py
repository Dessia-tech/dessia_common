from dessia_common.models.workflows import workflow_
import unittest
from parameterized import parameterized

parameter_input = workflow_.blocks[0].inputs[0]
integer_input = workflow_.nonblock_variables[0]
string_input = workflow_.nonblock_variables[1]
n_solutions = 100
input_values = {workflow_.input_index(parameter_input): 5,
                workflow_.input_index(integer_input): n_solutions,
                workflow_.input_index(string_input): "Test"}
workflow_run = workflow_.run(input_values=input_values, verbose=True, name='Dev Objects')


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
        variables_match = workflow_.match_variables(True)
        self.assertIn(variable, variables_match)
        self.assertEqual(variables_match[variable], expected_match)

    @parameterized.expand([
        ("#/values/1", "Test"),
        ("#/values/0", n_solutions)
    ])
    def test_getattr(self, path, expected_result):
        self.assertEqual(workflow_run._get_from_path(path), expected_result)

    def test_failing_getattr(self):
        self.assertRaises(AttributeError, workflow_run._get_from_path("#/values/1/0"))

    def test_check_platform(self):
        workflow_._check_platform()
        workflow_run._check_platform()

        copied_workflow_run = workflow_run.copy()
        copied_workflow_run._check_platform()

    def test_arguments(self):
        arguments = workflow_.dict_to_arguments(input_values, "run")
        self.assertDictEqual(arguments, {'input_values': {0: 5, 1: None, 2: 3, 3: 100, 4: 'Test'}, 'name': None})
