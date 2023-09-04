import unittest
from parameterized import parameterized
from dessia_common.forms import StandaloneObject


class TestComputationFromDecorators(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = StandaloneObject._display_settings_from_decorators()

    def test_number_of_settings(self):
        self.assertEqual(len(self.settings), 6)

    @parameterized.expand([
        (0, "Graph 2D"),
        (1, "Multiplot"),
        (2, "Parallel Plot"),
        (3, "2D View"),
        (4, "Scatter Plot"),
        (5, "Markdown")
    ])
    def test_decorators(self, index, expected_selector):
        self.assertEqual(self.settings[index].selector, expected_selector)
