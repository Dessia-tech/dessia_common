from dessia_common.models.workflows.forms_workflow import workflow_


copied_workflow = workflow_.copy()


assert len(workflow_.pipes) == len(copied_workflow.pipes)
assert len(workflow_.nonblock_variables) == len(copied_workflow.nonblock_variables)

assert all([p.input_variable.name == cp.input_variable.name for p, cp in zip(workflow_.pipes, copied_workflow.pipes)])

copied_workflow._check_platform()

print("script 'pipes.py' has passed")
