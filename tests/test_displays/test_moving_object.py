from dessia_common.forms import MovingStandaloneObject
import unittest
from parameterized import parameterized


class TestMovingObject(unittest.TestCase):
    def setUp(self) -> None:
        self.mso = MovingStandaloneObject(origin=0, name="Moving Test")
        self.displays = self.mso._displays()

    def test_viability(self):
        self.mso._check_platform()

    def test_length(self):
        self.assertEqual(len(self.displays), 3)

    @parameterized.expand([
        (0, "markdown"),
        (1, "tree"),
        (2, "babylon_data"),
    ])
    def test_decorators(self, index, expected_type):
        self.assertEqual(self.displays[index]["type_"], expected_type)
