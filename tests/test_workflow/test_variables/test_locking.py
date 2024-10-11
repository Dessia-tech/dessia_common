from dessia_common.workflow.core import Variable
from dessia_common.schemas.core import UNDEFINED
import unittest
from parameterized import parameterized

BARE_VARIABLE = Variable()

VARIABLE_WITH_DEFAULT_VALUE = Variable(default_value=1)

LOCKED_VARIABLE = Variable()
LOCKED_VARIABLE.lock(2)

LOCKED_VARIABLE_WITH_DEFAULT_VALUE = Variable(default_value=3)
LOCKED_VARIABLE_WITH_DEFAULT_VALUE.lock()

OVERWRITTEN_LOCKED_VARIABLE = Variable(default_value=4)
OVERWRITTEN_LOCKED_VARIABLE.lock(5)


class TestVariableFeatures(unittest.TestCase):

    @parameterized.expand([
        (BARE_VARIABLE, False, UNDEFINED),
        (VARIABLE_WITH_DEFAULT_VALUE, False, 1),
        (LOCKED_VARIABLE, True, 2),
        (LOCKED_VARIABLE_WITH_DEFAULT_VALUE, True, 3),
        (OVERWRITTEN_LOCKED_VARIABLE, True, 5)

    ])
    def test_default_values(self, variable, is_locked, expected_value):
        self.assertEqual(variable.locked, is_locked)
        self.assertEqual(variable.default_value, expected_value)

    def test_forbidden_behavior(self):
        variable = Variable()
        self.assertRaises(ValueError, variable.lock)

