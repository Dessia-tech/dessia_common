import unittest
from parameterized import parameterized
from dessia_common.forms import StandaloneObject
from dessia_common.workflow.blocks import ModelMethod
from dessia_common.typings import MethodType

MODEL_METHOD = ModelMethod(method_type=MethodType(class_=StandaloneObject, name="count_until"))
EXPECTED_DESCRIPTIONS = [
    "Standalone Object for testing purpose.",
    "Duration of the method in s",
    "Whether the computation should raise an error or not at the end"
]

class TestBlockDocstrings(unittest.TestCase):

    @parameterized.expand([
        (MODEL_METHOD, 3),
    ])
    def test_model_method(self, block, expected_length: int):
        parsed_docstring = block._docstring()
        self.assertEqual(len(parsed_docstring), expected_length)
        for i, expected_description in enumerate(EXPECTED_DESCRIPTIONS):
            input_ = block.inputs[i]
            self.assertEqual(parsed_docstring[input_]["desc"], expected_description)


if __name__ == '__main__':
    unittest.main()
