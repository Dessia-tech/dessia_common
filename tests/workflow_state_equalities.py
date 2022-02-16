from dessia_common.models.workflows.forms_workflow import workflow_
from dessia_common.models.workflows.wokflow_exports import workflow_export

workflow_ = workflow_export

print("============== Blank Start ==============")
ws_empty = workflow_.start_run({})

copied_ws_empty = ws_empty.copy()
print(copied_ws_empty == ws_empty)
assert copied_ws_empty == ws_empty

dict_empty = copied_ws_empty.to_dict()

print("============== Start with values ==============")
workflow_state = workflow_.start_run({0: 2, 4: 3})
partial_ws = workflow_state.copy()
print(partial_ws == workflow_state)
assert partial_ws == workflow_state

partial_dict = partial_ws.to_dict()

print("============== Next Block ==============")
partial_ws.evaluate_next_block()
c_partial_ws = partial_ws.copy()
print(c_partial_ws == partial_ws)
assert c_partial_ws == partial_ws

print("============== Continue Run ==============")
to_continue_ws = workflow_.start_run({0: 1, 4: 4})
to_continue_ws.continue_run()
continued_ws = to_continue_ws.copy()
print(continued_ws == to_continue_ws)
assert continued_ws == to_continue_ws




