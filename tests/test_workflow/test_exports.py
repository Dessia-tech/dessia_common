from dessia_common.models.workflows.workflow_exports import workflow_export_run
from dessia_common.files import StringFile, BinaryFile
import unittest
from parameterized import parameterized


class TestWorkflowExports(unittest.TestCase):
    def setUp(self) -> None:
        self.export_formats = workflow_export_run._export_formats()

    def test_length(self):
        self.assertEqual(len(self.export_formats), 5)

    @parameterized.expand([
        (0, "json"),
        (1, "Export JSON (6)"),
        (2, "Zip (7)"),
        (3, "Export XLSX (8)")
    ])
    def test_selectors(self, index, expected_selector):
        format_ = self.export_formats[index]
        self.assertEqual(format_.selector, expected_selector)

    @parameterized.expand([
        (0, "json"),
        (1, "json"),
        (2, "zip"),
        (3, "xlsx")
    ])
    def test_extension(self, index, expected_extension):
        format_ = self.export_formats[index]
        self.assertEqual(format_.extension, expected_extension)

    @parameterized.expand([
        (0, "save_to_stream"),
        (1, "export"),
        (2, "export"),
        (3, "export")
    ])
    def test_method_names(self, index, expected_name):
        format_ = self.export_formats[index]
        self.assertEqual(format_.method_name, expected_name)

    @parameterized.expand([
        (0, True),
        (1, True),
        (2, False),
        (3, False)
    ])
    def test_text(self, index, expected_bool):
        format_ = self.export_formats[index]
        self.assertEqual(format_.text, expected_bool)

    @parameterized.expand([
        (0, ""),
        (1, "export_json"),
        (2, "archive"),
        (3, "export_xlsx")
    ])
    def test_export_names(self, index, expected_name):
        format_ = self.export_formats[index]
        self.assertEqual(format_.export_name, expected_name)

    @parameterized.expand([
        (0, {}),
        (1, {"block_index": 6}),
        (2, {"block_index": 7}),
        (3, {"block_index": 8})
    ])
    def test_args(self, index, expected_args):
        format_ = self.export_formats[index]
        self.assertDictEqual(format_.args, expected_args)

    @parameterized.expand([
        (1, StringFile),
        (2, BinaryFile),
        (3, BinaryFile)
    ])
    def test_export(self, index, file_type):
        format_ = self.export_formats[index]
        block_index = format_.args["block_index"]
        stream = file_type()
        workflow_export_run.export(stream, block_index)
        self.assertNotEquals(stream.getvalue(), "")
