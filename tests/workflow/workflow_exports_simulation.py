from dessia_common.models.workflows.wokflow_exports import workflow_export_run
from dessia_common.files import StringFile, BinaryFile

export_formats = workflow_export_run._export_formats()
assert len(export_formats) == 3
json_export = export_formats[0]
assert json_export.selector == "Export JSON (6)"
assert json_export.extension == "json"
assert json_export.method_name == "export"
assert json_export.text is True
assert json_export.export_name == "archive"
assert json_export.args == {"block_index": 6}

archive_export = export_formats[1]
assert archive_export.selector == "Zip (7)"
assert archive_export.extension == "zip"
assert archive_export.method_name == "export"
assert archive_export.text is False
assert archive_export.method_name == "archive"
assert archive_export.args == {'block_index': 7}

json_stream = StringFile()
xlsx_stream = BinaryFile()
zip_stream = BinaryFile()
a = workflow_export_run.export(json_stream, 6)
workflow_export_run.export(zip_stream, 7)
workflow_export_run.export(xlsx_stream, 8)

print("workflow_exports_simulation.py has passed")
