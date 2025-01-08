import unittest
from parameterized import parameterized
from dessia_common.workflow.core import Workflow
from dessia_common.workflow.blocks import InstantiateModel
from dessia_common.core import DessiaObject


class TestHandling(unittest.TestCase):

    def setUp(self):
        block = InstantiateModel(model_class=DessiaObject)
        self.workflow = Workflow(blocks=[block], pipes=[], output=block.outputs[0])
        self.workflow.insert_step("A")
        self.workflow.insert_step("B")
        self.workflow.insert_step("C")

    def test_init(self):
        self.assertEqual(len(self.workflow.steps), 3)

    @parameterized.expand([
        (0, ["B", "C"]),
        (1, ["A", "C"]),
        (2, ["A", "B"]),
    ])
    def test_remove_one(self, index: int, expected_labels: list[int]):
        self.workflow.remove_step(index)
        self.assertEqual(len(self.workflow.steps), 2)
        ordered_labels = [s.label for s in self.workflow.steps]
        self.assertListEqual(ordered_labels, expected_labels)

    @parameterized.expand([
        ([2, 1], ["A"]),
        ([2, 0], ["B"]),
        ([1, 1], ["A"]),
        ([1, 0], ["C"]),
        ([0, 1], ["B"]),
        ([0, 0], ["C"]),
        ([2, 1, 0], []),
        ([2, 0, 0], []),
        ([1, 1, 0], []),
        ([1, 0, 0], []),
        ([0, 1, 0], []),
        ([0, 0, 0], [])
    ])
    def test_remove_many(self, indices: list[int], expected_labels: list[str]):
        """ Base steps are ['A', 'B', 'C'] """
        for i, index in enumerate(indices):
            self.workflow.remove_step(index)
        self.assertEqual(len(self.workflow.steps), len(expected_labels))
        ordered_labels = [s.label for s in self.workflow.steps]
        self.assertListEqual(ordered_labels, expected_labels)

    def test_remove_error(self):
        self.workflow.remove_step(0)
        self.workflow.remove_step(0)
        self.assertRaises(IndexError, self.workflow.remove_step, 1)
        self.workflow.remove_step(0)
        self.assertRaises(IndexError, self.workflow.remove_step, 0)
        self.assertRaises(TypeError, self.workflow.remove_step, None)

    def test_rename(self):
        new_label = "Renamed A"
        self.workflow.rename_step(0, new_label)
        self.assertEqual(self.workflow.steps[0].label, new_label)

    def test_reset(self):
        self.workflow.reset_steps()
        self.assertEqual(len(self.workflow.steps), 0)


class TestInsertion(unittest.TestCase):

    def setUp(self):
        block = InstantiateModel(model_class=DessiaObject)
        self.workflow = Workflow(blocks=[block], pipes=[], output=block.outputs[0])

    @parameterized.expand([
        (0, "A"),
        (1, "B"),
        (1000, "C"),
        (None, "D")
    ])
    def test_insert_one(self, index: int, label: str):
        self.workflow.insert_step(label=label, index=index)
        self.assertEqual(len(self.workflow.steps), 1)
        self.assertEqual(self.workflow.steps[0].label, label)

    @parameterized.expand([
        ([None, None, None], ["A", "B", "C"], ["A", "B", "C"]),
        ([0, 1, 2, 3], ["A", "B", "C", "C"], ["A", "B", "C", "C"]),
        ([1, 2, 3], ["A", "B", "C"], ["A", "B", "C"]),
        (
                [3, 2, 1, 1000, 0, None, 6, 4],
                ["3", "2", "1", "1000", "0", "None", "6", "4"],
                ["0", "3", "1", "2", "4", "1000", "None", "6"]
        ),
    ])
    def test_insert_many(self, indices: list[int], labels: list[str], expected_order: list[str]):
        for i, index in enumerate(indices):
            self.workflow.insert_step(label=labels[i], index=index)
        self.assertEqual(len(indices), len(self.workflow.steps))
        ordered_labels = [s.label for s in self.workflow.steps]
        self.assertListEqual(ordered_labels, expected_order)


class TestReorder(unittest.TestCase):

    def setUp(self):
        block = InstantiateModel(model_class=DessiaObject)
        self.workflow = Workflow(blocks=[block], pipes=[], output=block.outputs[0])
        self.workflow.insert_step(label="0")
        self.workflow.insert_step(label="1")
        self.workflow.insert_step(label="2")
        self.workflow.insert_step(label="3")
        self.workflow.insert_step(label="4")

    @parameterized.expand([
        ([0, 1, 2, 3, 4], ["0", "1", "2", "3", "4"]),
        ([4, 3, 2, 1, 0], ["4", "3", "2", "1", "0"]),
        ([2, 1, 0, 3, 4], ["2", "1", "0", "3", "4"]),
        ([1, 2, 0, 3, 4], ["1", "2", "0", "3", "4"]),
    ])
    def test_baseline(self, indices: list[int], expected_order: list[str]):
        self.workflow.reorder_steps(indices)
        ordered_labels = [s.label for s in self.workflow.steps]
        self.assertListEqual(ordered_labels, expected_order)

    @parameterized.expand([
        ([0, 1, 2, 3], ValueError),
        ([0, 3, 2, 1, 0], ValueError)
    ])
    def test_errors(self, indices: list[int], expected_exception: type[Exception]):
        self.assertRaises(expected_exception, self.workflow.reorder_steps, indices)
