import unittest
from parameterized import parameterized
from dessia_common.forms import StandaloneObject


class TestComputationFromDecorators(unittest.TestCase):
    @parameterized.expand([
        (0, "2DTest"),
        (1, "MDTest"),
        (2, "cad_display_method")
    ])
    def test_decorators(self, index, expected_selector):
        settings = StandaloneObject._display_settings_from_decorators()
        self.assertEqual(len(settings), 3)
        self.assertEqual(settings[index], expected_selector)
