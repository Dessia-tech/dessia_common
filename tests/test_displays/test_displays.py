import unittest
from dessia_common.forms import StandaloneObject


class TestDisplaySettingFailure(unittest.TestCase):
    def test_wrong_decorators(self):
        self.assertRaises(ValueError, StandaloneObject._display_settings_from_selector, "wrong_selector")
