from dessia_common.forms import DEF_SO

schemas = DEF_SO.method_schemas

assert len(schemas) == 6
assert set(DEF_SO._allowed_methods) == set(schemas.keys())

count_until_schema = schemas["count_until"]
assert count_until_schema["type"] == "object"
assert count_until_schema["required"] == ["0"]
assert count_until_schema["description"] == "Test long execution with a customizable duration."
assert len(count_until_schema["properties"]) == 2
assert count_until_schema["properties"]["0"] == {"title": "Duration", "editable": True, "python_typing": "float",
                                                 "description": "Duration of the method in s", "type": "number"}
assert count_until_schema["properties"]["1"] == {
    "title": "Raise Error", "editable": True, "type": "boolean", "default_value": False,
    "description": "Wether the computation should raise an error or not at the end", "python_typing": "bool"
}

print("script 'methods.py' has passed.")

