import unittest
from parameterized import parameterized
from dessia_common.forms import Generator
from dessia_common.workflow.core import Workflow, Pipe, Variable
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod
from dessia_common.typings import MethodType
from dessia_common.models.workflows.sandbox_workflow import (workflow, generator_block, generate_block, zip_block,
                                                             foreach_block, concat_block, setattr_block, import_block,
                                                             variable_0)


PARAMETER = generator_block.inputs[0]
NB_SOLUTIONS = generator_block.inputs[1]
MODELS = generator_block.inputs[2]
OVERWRITE_TUPLE = setattr_block.inputs[1]
FILENAME = zip_block.inputs[2]
DURATION = foreach_block.inputs[1]
RAISE_ERROR = foreach_block.inputs[2]
STREAM = import_block.inputs[0]
RESULT_NAME = variable_0


generator_block = InstantiateModel(model_class=Generator, name="Generator")


class TestUpstreamVariables(unittest.TestCase):
    @parameterized.expand([
        (generator_block.outputs[0], [PARAMETER, NB_SOLUTIONS, MODELS, RESULT_NAME]),
        (generate_block.outputs[0], [PARAMETER, NB_SOLUTIONS, MODELS, RESULT_NAME]),
        (generate_block.outputs[1], [PARAMETER, NB_SOLUTIONS, MODELS, RESULT_NAME]),
        (zip_block.outputs[0], [PARAMETER, NB_SOLUTIONS, MODELS, RESULT_NAME, FILENAME]),
        (foreach_block.outputs[0], [
            PARAMETER, NB_SOLUTIONS, MODELS, RESULT_NAME, DURATION, RAISE_ERROR, OVERWRITE_TUPLE, STREAM
        ]),
        (concat_block.outputs[0], [RESULT_NAME])
    ])
    def test_necessary_inputs(self, variable, expected_inputs):
        inputs = workflow.upstream_inputs(variable)
        self.assertEqual(len(inputs), len(expected_inputs))
        for input_ in expected_inputs:
            self.assertIn(input_, inputs)


class TestBlockUpstreams(unittest.TestCase):

    def setUp(self):
        self.generator = InstantiateModel(model_class=Generator, name="Generator")
        self.generate = ModelMethod(method_type=MethodType(Generator, 'generate'), name="Generate")
        blocks = [self.generator, self.generate]
        pipes = [Pipe(self.generator.outputs[0], self.generate.inputs[0])]
        self.workflow = Workflow(blocks=blocks, pipes=pipes, output=self.generate.outputs[0])

    def test_wireness(self):
        upstream_variables = self.workflow.block_upstream_variables(self.generate)
        wired_inputs = upstream_variables["wired"]
        self.assertEqual(len(wired_inputs), 1)
        self.assertIs(wired_inputs[0], self.generator.outputs[0])

    def test_availability(self):
        upstream_variables = self.workflow.block_upstream_variables(self.generator)
        available_inputs = upstream_variables["available"]
        self.assertEqual(len(available_inputs), 4)
        for input_ in self.generator.inputs:
            self.assertIn(input_, available_inputs)


class TestNonBlockUpstreams(unittest.TestCase):

    def setUp(self):
        self.generator = InstantiateModel(model_class=Generator, name="Generator")
        self.generate = ModelMethod(method_type=MethodType(Generator, 'generate'), name="Generate")
        self.result_name = Variable(name='_result_name_', label='Result Name', type_=str)
        blocks = [self.generator, self.generate]
        pipes = [
            Pipe(self.result_name, self.generator.inputs[3]),
            Pipe(self.generator.outputs[0], self.generate.inputs[0])
        ]
        self.workflow = Workflow(blocks=blocks, pipes=pipes, output=self.generate.outputs[0])

    def test_nonblock(self):
        upstream_variables = self.workflow.block_upstream_variables(self.generator)
        available_inputs = upstream_variables["available"]
        nonblock_inputs = upstream_variables["nonblock"]
        self.assertEqual(len(available_inputs), 3)
        self.assertEqual(len(nonblock_inputs), 1)
        for input_ in self.generator.inputs[:3]:
            self.assertIn(input_, available_inputs)
        self.assertIn(self.result_name, nonblock_inputs)

    def test_locked_input(self):
        self.generator.inputs[0].lock(2)
        upstream_variables = self.workflow.block_upstream_variables(self.generator)
        available_inputs = upstream_variables["available"]
        nonblock_inputs = upstream_variables["nonblock"]
        locked_inputs = upstream_variables["locked"]
        self.assertEqual(len(available_inputs), 2)
        self.assertEqual(len(nonblock_inputs), 1)
        self.assertEqual(len(locked_inputs), 1)
        self.assertIn(self.generator.inputs[0], locked_inputs)
        self.assertIn(self.generator.inputs[1], available_inputs)
        self.assertIn(self.generator.inputs[2], available_inputs)
        self.assertIn(self.result_name, nonblock_inputs)

    def test_locked_nbv(self):
        self.result_name.lock("My Locked Name")
        upstream_variables = self.workflow.block_upstream_variables(self.generator)
        available_inputs = upstream_variables["available"]
        locked_inputs = upstream_variables["locked"]
        self.assertEqual(len(available_inputs), 3)
        self.assertEqual(len(locked_inputs), 1)
        self.assertIn(self.generator.inputs[0], available_inputs)
        self.assertIn(self.generator.inputs[1], available_inputs)
        self.assertIn(self.generator.inputs[2], available_inputs)
        self.assertIn(self.result_name, locked_inputs)
