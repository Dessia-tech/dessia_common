from dessia_common.models.workflows.workflow_displays import workflow
from dessia_common.utils.types import is_jsonable
import unittest
from parameterized import parameterized


class TestWorkflowDisplays(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow_run = workflow.run(input_values={0: 1})
        self.display_settings = self.workflow_run.display_settings()

    @parameterized.expand([
        (0, "Documentation"),
        (1, "Scatter Plot"),
        (2, "Markdown")
    ])
    def test_selectors(self, index, expected_selector):
        setting = self.display_settings[index]
        self.assertEqual(setting.selector, expected_selector)

    @parameterized.expand([
        ("Documentation", "markdown"),
        ("Scatter Plot", "plot_data"),
        ("Markdown", "markdown")
    ])
    def test_types(self, selector, expected_type):
        setting = self.workflow_run._display_settings_from_selector(selector)
        self.assertEqual(setting.type, expected_type)

    def test_is_jsonable(self):
        for setting in self.display_settings:
            display = self.workflow_run._display_from_selector(setting.selector)
            self.assertTrue(is_jsonable(display.data))

    @parameterized.expand([
        ("Documentation", str),
        ("Scatter Plot", dict),
        ("Markdown", str)
    ])
    def test_data_types(self, selector, expected_type):
        display = self.workflow_run._display_from_selector(selector)
        self.assertIsInstance(display.data, expected_type)

    @parameterized.expand([
        (True,),
        (False,)
    ])
    def test_default_displays(self, block_by_default: bool):
        self.assertFalse(workflow._display_settings_from_selector("Documentation").load_by_default)
        self.assertTrue(workflow._display_settings_from_selector("Tasks").load_by_default)

        workflow.blocks[1].load_by_default = block_by_default
        settings = self.workflow_run._display_settings_from_selector("Documentation")
        self.assertEqual(settings.load_by_default, not block_by_default)


if __name__ == '__main__':
    unittest.main()
