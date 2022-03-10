from dessia_common.workflow import WorkflowRun, Workflow
import json

from dessia_common.models.workflows import workflow_

# Check Workflow
serialized_workflow = workflow_.to_dict(use_pointers=False)
deserialized_workflow = Workflow.dict_to_object(dict_=serialized_workflow)

assert hash(workflow_) == hash(deserialized_workflow)
parameter_input = workflow_.blocks[0].inputs[0]
integer_input = workflow_.nonblock_variables[0]
string_input = workflow_.nonblock_variables[1]
input_values = {workflow_.input_index(parameter_input): 5,
                workflow_.input_index(integer_input): 2,
                workflow_.input_index(string_input): "Test"}
workflow_run = workflow_.run(input_values=input_values, verbose=True, name='Dev Objects')

# Check WorkflowRun
# Assert to_dict, dict_to_object, hashes, eqs
dict_ = workflow_run.to_dict(use_pointers=False)
object_ = WorkflowRun.dict_to_object(dict_=dict_)

assert hash(workflow_run) == hash(object_)

# Assert deserialization
demo_workflow_dict = workflow_.to_dict(use_pointers=False)
demo_workflow_json = json.dumps(demo_workflow_dict)
dict_from_json = json.loads(demo_workflow_json)
deserialized_demo_workflow = Workflow.dict_to_object(dict_from_json)
assert workflow_ == deserialized_demo_workflow


# Check WorkflowState
workflow_state = workflow_.start_run({})
workflow_state.add_block_input_values(0, input_values)

workflow_._check_platform()
# workflow_run._check_platform()

arguments = workflow_.dict_to_arguments(input_values, 'run')
