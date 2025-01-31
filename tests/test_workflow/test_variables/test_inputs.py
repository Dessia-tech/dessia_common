import unittest
from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod
from dessia_common.typings import MethodType
from dessia_common.forms import Generator


class TestVariableFeatures(unittest.TestCase):

    def setUp(self):
        generator = InstantiateModel(model_class=Generator, name="Generator")
        generate_method = MethodType(class_=Generator, name="generate")
        generate = ModelMethod(method_type=generate_method, name="Generate")

        blocks = [generator, generate]
        pipes = [Pipe(generator.outputs[0], generate.inputs[0])]

        self.workflow = Workflow(blocks=blocks, pipes=pipes, output=generate.outputs[0])

    def test_bare_run(self):
        self.assertRaises(ValueError, self.workflow.run)

    def test_no_lock(self):
        user_values = {0: 1}
        input_values = self.workflow.input_values(user_values)
        self.assertDictEqual(input_values, {0: 1, 1: 25, 2: None, 3: ""})
        self.workflow.run(user_values)

    def test_lock(self):
        self.workflow.inputs[0].lock(2)
        input_values = self.workflow.input_values()
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: ""})
        self.workflow.run()

    def test_locked_overwrite(self):
        self.workflow.inputs[0].lock(2)
        input_values = self.workflow.input_values({0: 3})
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: ""})
        self.workflow.run()

    def test_default_overwrite(self):
        self.workflow.inputs[0].lock(2)
        input_values = self.workflow.input_values({1: 10})
        self.assertDictEqual(input_values, {0: 2, 1: 10, 2: None, 3: ""})
        self.workflow.run()

    def test_locked_default(self):
        self.workflow.inputs[0].lock(2)
        self.workflow.inputs[1].lock()
        input_values = self.workflow.input_values()
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: ""})
        self.workflow.run()

    def test_locked_default_overwrite(self):
        self.workflow.inputs[0].lock(2)
        self.workflow.inputs[1].lock()
        input_values = self.workflow.input_values({1: 10})
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: ""})
        self.workflow.run()

    def test_unlock_keep_default(self):
        self.workflow.inputs[0].lock(2)
        self.workflow.inputs[3].lock("My locked variable")
        self.workflow.inputs[3].unlock()
        input_values = self.workflow.input_values()
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: "My locked variable"})
        self.workflow.run()

    def test_unlock_keep_default_overwrite(self):
        self.workflow.inputs[0].lock(2)
        self.workflow.inputs[3].lock("My locked variable")
        self.workflow.inputs[3].unlock()
        input_values = self.workflow.input_values({3: "Not Default"})
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: "Not Default"})
        self.workflow.run()

    def test_unlock_dispose_default_fails(self):
        self.workflow.inputs[0].lock(2)
        self.workflow.inputs[3].lock("My locked variable")
        self.workflow.inputs[3].unlock(False)
        input_values = self.workflow.input_values()
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None})
        self.assertRaises(ValueError, self.workflow.run)

    def test_unlock_dispose_default_overwrite(self):
        user_values = {3: "Not Default"}
        self.workflow.inputs[0].lock(2)
        self.workflow.inputs[3].lock("My locked variable")
        self.workflow.inputs[3].unlock(False)

        input_values = self.workflow.input_values(user_values)
        self.assertDictEqual(input_values, {0: 2, 1: 25, 2: None, 3: "Not Default"})
        self.workflow.run(user_values)

