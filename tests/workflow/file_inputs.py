from dessia_common.models.workflows.workflow_from_file_input import workflow_
from dessia_common.files import StringFile

with open('../../dessia_common/models/data/seed_file.csv') as stream:
    string_file = StringFile("test_file_input")
    string_file.write(stream.read())
    inputs = {0: string_file}
    workflow_run = workflow_.run(input_values=inputs)
    assert workflow_run.output_value.intarg == 7

assert workflow_._method_jsonschemas == {
    'run': {
        'definitions': {},
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'type': 'object',
        'required': ['0'],
        'properties': {
            '0': {
                'title': 'Class Method - stream', 'editable': True, 'order': 1, 'description': '',
                'python_typing': 'dessia_common.files.StringFile', 'type': 'text', 'is_file': True
            },
            '2': {
                'type': 'string', 'title': 'WorkflowRun Name', 'editable': True, 'order': 0,
                'description': 'Name for the resulting WorkflowRun', 'default_value': '',
                'python_typing': 'builtins.str'
            }
        },
        'classes': ['dessia_common.workflow.Workflow'],
        'method': True,
        'python_typing': 'dessia_common.typings.MethodType'
    },
    'start_run': {
        'definitions': {},
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'type': 'object',
        'required': [],
        'properties': {
            '0': {
                'title': 'Class Method - stream', 'editable': True, 'order': 1, 'description': '',
                'python_typing': 'dessia_common.files.StringFile', 'type': 'text', 'is_file': True
            },
            '2': {
                'type': 'string', 'title': 'WorkflowRun Name', 'editable': True, 'order': 0,
                'description': 'Name for the resulting WorkflowRun', 'default_value': '',
                'python_typing': 'builtins.str'
            }
        },
        'classes': ['dessia_common.workflow.Workflow'],
        'method': True,
        'python_typing': 'dessia_common.typings.MethodType'
    }
}


