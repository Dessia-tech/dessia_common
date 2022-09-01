import unittest

from dessia_common.forms import StandaloneObjectWithDefaultValues, StandaloneObject
from dessia_common.typings import MethodType
from dessia_common.workflow import InstantiateModel, Workflow, WorkflowError, Pipe, ModelMethod


class WorkflowUnitTest(unittest.TestCase):
    block_0 = InstantiateModel(model_class=StandaloneObjectWithDefaultValues, name='Instantiate SOWDV')
    block_0_bis = InstantiateModel(model_class=StandaloneObjectWithDefaultValues, name='Instantiate SOWDV')
    block_1 = ModelMethod(
        method_type=MethodType(class_=StandaloneObject, name='add_float'),
        name="StandaloneObject add float"
    )


class WorkflowTests(WorkflowUnitTest):

    def test_output_in_init(self):
        # Should not raise an error
        Workflow(
            blocks=[self.block_0, self.block_1],
            pipes=[],
            output=self.block_0.outputs[0]
        )

        # Following asserts are OK iff Workflow.__init__ raises an error
        # as output is not valid
        self.assertRaises(
            WorkflowError,
            Workflow,
            name="SOWDV",
            blocks=[self.block_0, self.block_1],
            pipes=[],
            output=self.block_0.inputs[0]
        )
        self.assertRaises(
            WorkflowError,
            Workflow,
            blocks=[self.block_1],
            pipes=[],
            output=self.block_0.outputs[0]
        )


class WorkflowDataEq(WorkflowUnitTest):

    def test_valid_example(self):
        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        self.assertTrue(wf1._data_eq(wf1))
        self.assertTrue(wf1._data_eq(wf2))
        self.assertTrue(wf2._data_eq(wf1))

    def test_output(self):
        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_1.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_1.outputs[1])
        self.assertFalse(wf1._data_eq(wf2))

    def test_blocks(self):
        wf1 = Workflow(blocks=[self.block_0], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        self.assertFalse(wf1._data_eq(wf2))

        wf1 = Workflow(blocks=[self.block_0, self.block_0], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        self.assertFalse(wf1._data_eq(wf2))

        wf1 = Workflow(blocks=[self.block_0], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0_bis], pipes=[], output=self.block_0_bis.outputs[0])
        self.assertTrue(wf1._data_eq(wf2))

    # # NOT IMPLEMENTED YET
    # def test_self.block_order(self):
    #     workflow1 = Workflow(blocks=[self.block_0, self.block_1],pipes=[],output=instantiate.outputs[0])
    #     workflow2 = Workflow(blocks=[self.block_1, self.block_0],pipes=[],output=instantiate.outputs[0])
    #     self.assertTrue(workflow1._data_eq(workflow2), expected_value)

    def test_pipes(self):
        pipe_0 = Pipe(self.block_0.outputs[0], self.block_1.inputs[0])
        pipe_0_bis = Pipe(self.block_0.outputs[0], self.block_1.inputs[0])
        pipe_1 = Pipe(self.block_0.outputs[0], self.block_1.inputs[1])

        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0], output=self.block_0.outputs[0])
        self.assertTrue(wf1._data_eq(wf2))

        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0_bis], output=self.block_0.outputs[0])
        self.assertTrue(wf1._data_eq(wf2))

        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0], output=self.block_0.outputs[0])
        self.assertFalse(wf1._data_eq(wf2))

        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_1], output=self.block_0.outputs[0])
        self.assertFalse(wf1._data_eq(wf2))

    def test_pipe_order(self):
        pipe_0 = Pipe(self.block_0.outputs[0], self.block_1.inputs[0])
        pipe_1 = Pipe(self.block_0.outputs[0], self.block_1.inputs[1])
        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_0, pipe_1], output=self.block_1.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[pipe_1, pipe_0], output=self.block_1.outputs[0])
        self.assertTrue(wf1._data_eq(wf2))

    def test_imposed_variable_values(self):
        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])

        wf1.imposed_variable_values[self.block_0.inputs[5]] = 42
        self.assertFalse(wf1._data_eq(wf2))

        wf2.imposed_variable_values[self.block_0.inputs[5]] = 17
        self.assertFalse(wf1._data_eq(wf2))

        wf2.imposed_variable_values[self.block_0.inputs[5]] = 42
        self.assertTrue(wf1._data_eq(wf2))

    #     test non-builtins variables : NOT IMPLEMENTED YET.

    def test_imposed_variable_value_order(self):
        wf1 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])
        wf2 = Workflow(blocks=[self.block_0, self.block_1], pipes=[], output=self.block_0.outputs[0])

        wf1.imposed_variable_values[self.block_0.inputs[5]] = 42
        wf1.imposed_variable_values[self.block_0.inputs[6]] = "my_test"

        wf2.imposed_variable_values[self.block_0.inputs[6]] = "my_test"
        wf2.imposed_variable_values[self.block_0.inputs[5]] = 42
        self.assertTrue(wf1._data_eq(wf2))

    # def test_non_self.block_variables(self):
    #     blocks = [self.block_0, self.block_1]
    #     output = self.block_0.outputs[0]
    #
    #     pipe1 = Pipe(self.block_0.outputs[0], self.block_1.inputs[1])
    #     pipe2 = Pipe(TypedVariable(type_=float), self.block_1.inputs[1])
    #     wf1 = Workflow(blocks=blocks, pipes=[pipe1], output=output)
    #     wf2 = Workflow(blocks=blocks, pipes=[pipe2], output=output)
    #     self.assertFalse(wf1._data_eq(wf2))
    #
    #     pipe1 = Pipe(TypedVariable(type_=float), self.block_1.inputs[1])
    #     pipe2 = Pipe(TypedVariable(type_=float), self.block_1.inputs[1])
    #     wf1 = Workflow(blocks=blocks, pipes=[pipe1], output=output)
    #     wf2 = Workflow(blocks=blocks, pipes=[pipe2], output=output)
    #     self.assertTrue(wf1._data_eq(wf2))
    #
    #     pipe1 = Pipe(TypedVariable(type_=float), self.block_1.inputs[1])
    #     pipe2 = Pipe(TypedVariableWithDefaultValue(type_=float, default_value=2.7), self.block_1.inputs[1])
    #     wf1 = Workflow(blocks=blocks, pipes=[pipe1], output=output)
    #     wf2 = Workflow(blocks=blocks, pipes=[pipe2], output=output)
    #     self.assertFalse(wf1._data_eq(wf2))
    #
    #     pipe1 = Pipe(TypedVariableWithDefaultValue(type_=float, default_value=2.7), self.block_1.inputs[1])
    #     pipe2 = Pipe(TypedVariableWithDefaultValue(type_=float, default_value=3.4), self.block_1.inputs[1])
    #     wf1 = Workflow(blocks=blocks, pipes=[pipe1], output=output)
    #     wf2 = Workflow(blocks=blocks, pipes=[pipe2], output=output)
    #     self.assertFalse(wf1._data_eq(wf2))
    #
    #     pipe1 = Pipe(TypedVariableWithDefaultValue(type_=float, default_value=3.4), self.block_1.inputs[1])
    #     pipe2 = Pipe(TypedVariableWithDefaultValue(type_=float, default_value=3.4), self.block_1.inputs[1])
    #     wf1 = Workflow(blocks=blocks, pipes=[pipe1], output=output)
    #     wf2 = Workflow(blocks=blocks, pipes=[pipe2], output=output)
    #     self.assertTrue(wf1._data_eq(wf2))
