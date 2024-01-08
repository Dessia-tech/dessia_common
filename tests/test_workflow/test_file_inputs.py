import unittest

import importlib.resources
from dessia_common.models.workflows.workflow_from_file_input import workflow_
from dessia_common.files import StringFile


class WorkflowInputTests(unittest.TestCase):
    def setUp(self) -> None:
        ref = importlib.resources.files("dessia_common").joinpath("models/data/seed_file.csv")
        with ref.open("rb") as stream:
            self.stringfile = StringFile.from_stream(stream)

    def test_run_by_string_file_input(self):
        inputs = {0: self.stringfile}
        workflow_run = workflow_.run(input_values=inputs)
        self.assertEqual(workflow_run.output_value.standalone_subobject.intarg, 7)
        workflow_run._check_platform()

    def tearDown(self) -> None:
        self.stringfile.close()


if __name__ == '__main__':
    unittest.main()
