from dessia_common.models.workflows import workflow_

# Check Workflow
parameter_input = workflow_.blocks[0].inputs[0]
integer_input = workflow_.nonblock_variables[0]
string_input = workflow_.nonblock_variables[1]
n_solutions = 100
input_values = {workflow_.input_index(parameter_input): 5,
                workflow_.input_index(integer_input): n_solutions,
                workflow_.input_index(string_input): "Test"}
workflow_run = workflow_.run(input_values=input_values, verbose=True, name='Dev Objects')

