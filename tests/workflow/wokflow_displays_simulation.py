from dessia_common.models.workflows.workflow_displays import workflow
from dessia_common.utils.types import is_jsonable

workflow_run = workflow.run(input_values={0: 1})
display_settings = workflow_run.display_settings()
assert len(display_settings) == 5
assert display_settings[0].selector == "documentation"
assert display_settings[1].selector == "workflow"
assert display_settings[2].selector == "3D (1)"
assert display_settings[3].selector == "2D (2)"
assert display_settings[4].selector == "MD (3)"

assert display_settings[0].type == "markdown"
assert display_settings[1].type == "workflow"
assert display_settings[2].type == "cad"
assert display_settings[3].type == "plot_data"
assert display_settings[4].type == "markdown"

cad_do = workflow_run.block_display(1)[0]
pd_do = workflow_run.block_display(2)[0]
md_do = workflow_run.block_display(3)[0]

assert is_jsonable(cad_do.data)
assert is_jsonable(pd_do.data)
assert is_jsonable(md_do.data)

assert len(cad_do.data["meshes"]) == 2
assert len(pd_do.data) == 5
assert isinstance(md_do.data, str)
assert len(md_do.data) == 2520

print("script workflow_displays_simulation.py has passed")
