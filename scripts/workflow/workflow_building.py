from dessia_common.workflow.core import Workflow, TypedVariableWithDefaultValue, Pipe, NAME_VARIABLE
from dessia_common.workflow.blocks import InstantiateModel
from dessia_common.forms import Generator

# Empty Workflow with only the init computed name variable
empty_workflow = Workflow(blocks=[], pipes=[], output=None)
empty_workflow._check_platform()
assert len(empty_workflow.detached_variables) == 1
assert empty_workflow.detached_variables[0].name == "Result Name"
assert len(empty_workflow.nonblock_variables) == 0

empty_dict = empty_workflow.to_dict()
assert len(empty_dict["nonblock_variables"]) == 1
assert empty_dict["nonblock_variables"][0]["name"] == "Result Name"

# Workflow with one block and the detached init computed name variable
block = InstantiateModel(model_class=Generator, name="Generator")
lone_block_workflow = Workflow(blocks=[block], pipes=[], output=None)
lone_block_workflow._check_platform()
assert len(lone_block_workflow.detached_variables) == 1
assert lone_block_workflow.detached_variables[0].name == "Result Name"
assert len(lone_block_workflow.nonblock_variables) == 0
assert len(lone_block_workflow.inputs) == 4  # TODO Should the detached variable be seen as an input ?

lone_block_dict = lone_block_workflow.to_dict()
assert len(empty_dict["nonblock_variables"]) == 1  # Detached variables are serialized as nonblock
assert empty_dict["nonblock_variables"][0]["name"] == "Result Name"

# Workflow with wired block and name variable. Name var should be nonblock and not detached anymore
pipe_name = Pipe(input_variable=NAME_VARIABLE, output_variable=lone_block_workflow.inputs[3])
wired_workflow = Workflow(blocks=[block], pipes=[pipe_name], output=None)
wired_workflow._check_platform()
assert len(wired_workflow.detached_variables) == 0
assert len(wired_workflow.nonblock_variables) == 1
assert wired_workflow.nonblock_variables[0].name == "Result Name"
assert len(wired_workflow.inputs) == 4

wired_dict = wired_workflow.to_dict()
assert len(wired_dict["nonblock_variables"]) == 1
assert wired_dict["nonblock_variables"][0]["name"] == "Result Name"

# Workflow with an user variable added as a detached one
user_variable = TypedVariableWithDefaultValue(type_=int, default_value="2", name="User Parameter")
user_variable_workflow = Workflow(blocks=[block], pipes=[pipe_name], output=None, detached_variables=[user_variable])
user_variable_workflow._check_platform()
assert len(user_variable_workflow.detached_variables) == 1
assert user_variable_workflow.detached_variables[0].name == "User Parameter"
assert len(user_variable_workflow.nonblock_variables) == 1
assert user_variable_workflow.nonblock_variables[0].name == "Result Name"
assert len(user_variable_workflow.inputs) == 4

user_variable_dict = user_variable_workflow.to_dict()
assert len(user_variable_dict["nonblock_variables"]) == 2
assert user_variable_dict["nonblock_variables"][0]["name"] == "Result Name"
assert user_variable_dict["nonblock_variables"][1]["name"] == "User Parameter"

# Unwired Workflow with an user variable added as a detached one
double_variable_workflow = Workflow(blocks=[block], pipes=[], output=None, detached_variables=[user_variable])
double_variable_workflow._check_platform()
assert len(double_variable_workflow.detached_variables) == 2
assert double_variable_workflow.detached_variables[0].name == "Result Name"
assert double_variable_workflow.detached_variables[1].name == "User Parameter"
assert len(double_variable_workflow.nonblock_variables) == 0
assert len(double_variable_workflow.inputs) == 4

double_variable_dict = double_variable_workflow.to_dict()
assert len(double_variable_dict["nonblock_variables"]) == 2
assert double_variable_dict["nonblock_variables"][0]["name"] == "Result Name"
assert double_variable_dict["nonblock_variables"][1]["name"] == "User Parameter"

# Workflow with name variable that isn't connected and user variable is
pipe_var = Pipe(input_variable=user_variable, output_variable=block.inputs[0])
switched_user_variable_workflow = Workflow(blocks=[block], pipes=[pipe_var], output=None,
                                           detached_variables=[user_variable])
switched_user_variable_workflow._check_platform()
assert len(switched_user_variable_workflow.detached_variables) == 1
assert switched_user_variable_workflow.detached_variables[0].name == "Result Name"
assert len(switched_user_variable_workflow.nonblock_variables) == 1
assert switched_user_variable_workflow.nonblock_variables[0].name == "User Parameter"
assert len(switched_user_variable_workflow.inputs) == 4

switched_user_variable_dict = switched_user_variable_workflow.to_dict()
assert len(switched_user_variable_dict["nonblock_variables"]) == 2
assert switched_user_variable_dict["nonblock_variables"][0]["name"] == "User Parameter"
assert switched_user_variable_dict["nonblock_variables"][1]["name"] == "Result Name"

# Variante of last check with user variable not defined as detached variable
switched_user_variable_workflow = Workflow(blocks=[block], pipes=[pipe_var], output=None)
switched_user_variable_workflow._check_platform()
assert len(switched_user_variable_workflow.detached_variables) == 1
assert switched_user_variable_workflow.detached_variables[0].name == "Result Name"
assert len(switched_user_variable_workflow.nonblock_variables) == 1
assert switched_user_variable_workflow.nonblock_variables[0].name == "User Parameter"
assert len(switched_user_variable_workflow.inputs) == 4

switched_user_variable_dict = switched_user_variable_workflow.to_dict()
assert len(switched_user_variable_dict["nonblock_variables"]) == 2
assert switched_user_variable_dict["nonblock_variables"][0]["name"] == "User Parameter"
assert switched_user_variable_dict["nonblock_variables"][1]["name"] == "Result Name"

# Complete Workflow
another_user_variable = TypedVariableWithDefaultValue(type_=int, default_value=5, name="Number of Solutions")
pipe_numb = Pipe(input_variable=another_user_variable, output_variable=block.inputs[1])
complete_workflow = Workflow(blocks=[block], pipes=[pipe_name, pipe_var, pipe_numb],
                             output=block.outputs[0], detached_variables=[user_variable])
complete_workflow._check_platform()
assert len(complete_workflow.detached_variables) == 0
assert len(complete_workflow.nonblock_variables) == 3
assert complete_workflow.nonblock_variables[0].name == "Result Name"
assert complete_workflow.nonblock_variables[1].name == "User Parameter"
assert complete_workflow.nonblock_variables[2].name == "Number of Solutions"
assert len(complete_workflow.inputs) == 4

complete_dict = complete_workflow.to_dict()
assert len(complete_dict["nonblock_variables"]) == 3
assert complete_dict["nonblock_variables"][0]["name"] == "Result Name"
assert complete_dict["nonblock_variables"][1]["name"] == "User Parameter"
assert complete_dict["nonblock_variables"][2]["name"] == "Number of Solutions"


print("workflow_building.py script has passed")
