import unittest
from parameterized import parameterized
from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod
from dessia_common.forms import Generator, StandaloneObject
from dessia_common.typings import MethodType


class TestDisplays(unittest.TestCase):

    def setUp(self):
        generator = InstantiateModel(model_class=Generator, name="Generator")
        generate_method = MethodType(class_=Generator, name='generate')
        generate = ModelMethod(method_type=generate_method, name='Generate')
        count_method = MethodType(class_=StandaloneObject, name='count_until')
        count = ModelMethod(method_type=count_method, name="Count")
        blocks = [generator, generate, count]
        pipes = [Pipe(generator.outputs[0], generate.inputs[0]),
                 Pipe(generate.outputs[0], count.inputs[0])]
        self.workflow = Workflow(blocks=blocks, pipes=pipes, output=count.outputs[0])

    def test_displayable_outputs(self):
        self.assertEqual(len(self.workflow.displayable_outputs), 3)

    @parameterized.expand([
            (0, 4),
            (1, 4),
            (2, 6),
        ])
    def test_displayable_upstreams(self, output_index: int, expected_upstreams_length: int):
        output = self.workflow.displayable_outputs[output_index]
        upstreams = self.workflow.upstream_inputs(output)
        self.assertEqual(len(upstreams), expected_upstreams_length)
