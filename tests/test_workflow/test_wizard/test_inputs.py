import unittest
from parameterized import parameterized
from dessia_common.workflow.core import Workflow
from dessia_common.workflow.blocks import InstantiateModel
from dessia_common.forms import Generator


class BaseClass(unittest.TestCase):
    def setUp(self):
        block = InstantiateModel(model_class=Generator)
        self.workflow = Workflow(blocks=[block], pipes=[], output=block.outputs[0], name=self.__class__.__name__)

    def test_reset(self):
        self.workflow.reset_steps()
        self.assertEqual(len(self.workflow.steps), 0)
        self.assertEqual(len(self.workflow.inputs), len(self.workflow.spare_inputs))


class TestNoStep(BaseClass):

    def test_init(self):
        self.assertEqual(len(self.workflow.steps), 0)
        self.assertEqual(len(self.workflow.spare_inputs), len(self.workflow.inputs))

    def test_input_step_search(self):
        step = self.workflow.find_input_step(self.workflow.inputs[0])
        self.assertIsNone(step)

    def test_move_one(self):
        self.assertRaises(IndexError, self.workflow.change_input_step, input_index=0, new_step_index=0)
        self.assertEqual(len(self.workflow.spare_inputs), len(self.workflow.inputs))


class TestSingleStep(BaseClass):

    def setUp(self):
        super().setUp()
        self.workflow.insert_step("A")

    def test_init(self):
        self.assertEqual(len(self.workflow.steps[0].inputs), 0)
        self.assertEqual(len(self.workflow.spare_inputs), len(self.workflow.inputs))

    @parameterized.expand([
        (0,),
        (1,),
        (2,),
        (3,),
    ])
    def test_move_one(self, index: int):
        new_step_index = 0
        self.workflow.change_input_step(input_index=index, new_step_index=new_step_index)
        step = self.workflow.steps[new_step_index]
        self.assertEqual(len(step.inputs), 1)
        self.assertEqual(len(self.workflow.spare_inputs), 3)

    @parameterized.expand([
        ([0, 1],),
        ([1, 0],),
        ([0, 1, 2, 3],),
        ([3, 2, 1, 0],),
        ([2, 0, 3, 1],),
        ([2, 0, 1],),
    ])
    def test_move_many(self, indices: list[int]):
        expected_inputs_set = set([self.workflow.inputs[i] for i in indices])
        new_step_index = 0
        for i in indices:
            self.workflow.change_input_step(input_index=i, new_step_index=new_step_index)
        spare_set = set(self.workflow.spare_inputs)
        step_inputs_set = set(self.workflow.steps[new_step_index].inputs)
        self.assertEqual(step_inputs_set, expected_inputs_set)
        union = spare_set.union(step_inputs_set)
        self.assertSetEqual(union, set(self.workflow.inputs))

    def test_remove_inputs(self):
        self.workflow.change_input_step(input_index=0, new_step_index=0)
        self.workflow.remove_input_from_step(index=0)
        self.assertEqual(len(self.workflow.steps[0].inputs), 0)
        self.assertEqual(len(self.workflow.inputs), 4)


class TestMultipleSteps(BaseClass):

    def setUp(self):
        super().setUp()
        self.workflow.insert_step("A")
        self.workflow.insert_step("B")
        self.workflow.insert_step("C")

        self.workflow.change_input_step(input_index=0, new_step_index=0)
        self.workflow.change_input_step(input_index=1, new_step_index=2)
        self.workflow.change_input_step(input_index=3, new_step_index=2)

    def test_init(self):
        self.assertEqual(len(self.workflow.steps[0].inputs), 1)
        self.assertEqual(len(self.workflow.steps[1].inputs), 0)
        self.assertEqual(len(self.workflow.steps[2].inputs), 2)
        self.assertEqual(len(self.workflow.spare_inputs), 1)

    @parameterized.expand([
        (0, 0, [1, 0, 2], 1),
        (1, 0, [2, 0, 1], 1),
        (2, 0, [2, 0, 2], 0),
        (3, 2, [1, 0, 2], 1),
        (3, 1, [1, 1, 1], 1),
    ])
    def test_move_one(self, index: int, new_step_index: int, expected_step_lengths: list[int],
                      expected_spare_length: int):
        self.workflow.change_input_step(input_index=index, new_step_index=new_step_index)
        step_lengths = [len(s.inputs) for s in self.workflow.steps]
        self.assertListEqual(step_lengths, expected_step_lengths)
        self.assertEqual(len(self.workflow.spare_inputs), expected_spare_length)

    def test_remove_inputs(self):
        self.workflow.remove_input_from_step(index=0)
        self.assertEqual(len(self.workflow.steps[0].inputs), 0)
        self.assertEqual(len(self.workflow.spare_inputs), 2)


class TestReorder(BaseClass):

    def setUp(self):
        super().setUp()
        self.workflow.insert_step(label="A")
        self.workflow.change_input_step(input_index=0, new_step_index=0)
        self.workflow.change_input_step(input_index=3, new_step_index=0)

    @parameterized.expand([
        ([0, 3],),
        ([3, 0],)
    ])
    def test_nominal(self, order: list[int]):
        step_index = 0
        self.workflow.reorder_step_inputs(step_index=step_index, order=order)
        ordered_inputs = [self.workflow.input_index(i) for i in self.workflow.steps[step_index].inputs]
        self.assertListEqual(ordered_inputs, order)

    @parameterized.expand([
        ([4, 0],),
        ([1, 2, 3],),
        ([1000],),
        ([None],)
    ])
    def test_error(self, order: list[int]):
        self.assertRaises(ValueError, self.workflow.reorder_step_inputs, step_index=0, order=order)

