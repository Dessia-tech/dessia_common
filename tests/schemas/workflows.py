from dessia_common.models.workflows.forms_workflow import workflow_

ms = workflow_.method_schemas

run_schema = workflow_.method_schemas["run"]
start_schema = workflow_.method_schemas["start_run"]

assert run_schema["required"] == ["0", "3", "5"]
assert len(run_schema["properties"]) == 6
assert run_schema["classes"] == "dessia_common.workflow.core.Workflow"
assert run_schema["method"]
assert run_schema["type"] == "object"
assert run_schema["properties"]["2"] == {
    "title": "ForEach - binding Workflow Block - optimization Value", "editable": True, "python_typing": "int",
    "type": "number", "default_value": 3,
    "description": {
        "desc": "value that will be added to model's intarg attribute",
        "type_": "int",
        "annotation": "<class 'int'>"
    }
}

assert start_schema["required"] == []
assert len(run_schema["properties"]) == 6
assert start_schema["python_typing"] == "dessia_common.typings.MethodType"
assert start_schema["properties"]["4"] == {"title": "Shared Name", "editable": True, "description": None,
                                           "python_typing": "str", "type": "string", "default_value": "Shared Name"}

# Test run and start equalities apart from required
del run_schema["required"]
del start_schema["required"]
assert run_schema == start_schema

print("script workflows.py has passed.")
