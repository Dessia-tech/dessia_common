import pkg_resources
from dessia_common.models.workflows.workflow_from_file_input import workflow_
from dessia_common.files import StringFile


stream = StringFile.from_stream(pkg_resources.resource_stream('dessia_common', 'models/data/seed_file.csv'))

string_file = StringFile("test_file_input")
string_file.write(stream.read())
inputs = {0: string_file}
workflow_run = workflow_.run(input_values=inputs)
assert workflow_run.output_value.standalone_subobject.intarg == 7
workflow_run._check_platform()
