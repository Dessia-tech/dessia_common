from dessia_common.models.workflows.workflow_exports import workflow_export_run
from dessia_common.files import StringFile, BinaryFile
import unittest
from parameterized import parameterized


class TestWorkflowExports(unittest.TestCase):
    def setUp(self) -> None:
        self.export_formats = workflow_export_run._export_formats()

    def test_length(self):
        self.assertEqual(len(self.export_formats), 8)

    @parameterized.expand([
        (0, "json"),
        (1, "xlsx"),
        (2, "docx"),
        (3, "zip"),
        (4, "Export JSON (6)"),
        (5, "Zip (7)"),
        (6, "Export XLSX (8)")
    ])
    def test_selectors(self, index, expected_selector):
        format_ = self.export_formats[index]
        self.assertEqual(format_.selector, expected_selector)

    @parameterized.expand([
        (0, "json"),
        (1, "xlsx"),
        (2, "docx"),
        (3, "zip"),
        (4, "json"),
        (5, "zip"),
        (6, "xlsx")
    ])
    def test_extension(self, index, expected_extension):
        format_ = self.export_formats[index]
        self.assertEqual(format_.extension, expected_extension)

    @parameterized.expand([
        (0, "save_to_stream"),
        (1, "to_xlsx_stream"),
        (2, "to_docx_stream"),
        (3, "to_zip_stream"),
        (4, "export"),
        (5, "export"),
        (6, "export")
    ])
    def test_method_names(self, index, expected_name):
        format_ = self.export_formats[index]
        self.assertEqual(format_.method_name, expected_name)

    @parameterized.expand([
        (0, True),
        (1, False),
        (2, False),
        (3, False),
        (4, True),
        (5, False),
        (6, False)
    ])
    def test_text(self, index, expected_bool):
        format_ = self.export_formats[index]
        self.assertEqual(format_.text, expected_bool)

    @parameterized.expand([
        (0, ""),
        (1, ""),
        (2, ""),
        (3, ""),
        (4, "export_json"),
        (5, "archive"),
        (6, "export_xlsx")
    ])
    def test_export_names(self, index, expected_name):
        format_ = self.export_formats[index]
        self.assertEqual(format_.export_name, expected_name)

    @parameterized.expand([
        (0, {}),
        (1, {}),
        (2, {}),
        (3, {}),
        (4, {"block_index": 6}),
        (5, {"block_index": 7}),
        (6, {"block_index": 8})
    ])
    def test_args(self, index, expected_args):
        format_ = self.export_formats[index]
        self.assertDictEqual(format_.args, expected_args)

    @parameterized.expand([
        (4, StringFile),
        (5, BinaryFile),
        (6, BinaryFile)
    ])
    def test_export(self, index, file_type):
        format_ = self.export_formats[index]
        block_index = format_.args["block_index"]
        stream = file_type()
        workflow_export_run.export(stream, block_index)
        self.assertNotEquals(stream.getvalue(), "")
