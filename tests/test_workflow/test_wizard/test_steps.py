import unittest
from parameterized import parameterized
from dessia_common.workflow.core import Workflow
from dessia_common.workflow.blocks import InstantiateModel
from dessia_common.core import DessiaObject


class TestCrud(unittest.TestCase):

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
        self.workflow.insert_step(index=index, label=label)
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
            self.workflow.insert_step(index=index, label=labels[i])
        self.assertEqual(len(indices), len(self.workflow.steps))
        ordered_labels = [s.label for s in self.workflow.steps]
        self.assertListEqual(ordered_labels, expected_order)

    def test_remove_one(self):
        self.assertEqual(len(self.workflow.steps), 0)
        self.workflow.insert_step(0, "A")
        self.assertEqual(len(self.workflow.steps), 1)
        self.workflow.remove_step(0)
        self.assertEqual(len(self.workflow.steps), 0)

    @parameterized.expand([
        ([0], ["A", "C"]),
        ([1], ["B", "C"]),
        ([2], ["B", "A"]),
        ([2, 1], ["B"]),
        ([2, 0], ["A"]),
        ([1, 1], ["B"]),
        ([1, 0], ["C"]),
        ([0, 1], ["A"]),
        ([0, 0], ["C"]),
        ([2, 1, 0], []),
        ([2, 0, 0], []),
        ([1, 1, 0], []),
        ([1, 0, 0], []),
        ([0, 1, 0], []),
        ([0, 0, 0], [])
    ])
    def test_remove_many(self, indices: list[int], expected_labels: list[str]):
        """ Base steps are ['B', 'A', 'C'] """
        self.assertEqual(len(self.workflow.steps), 0)
        self.workflow.insert_step(0, "A")
        self.workflow.insert_step(0, "B")
        self.workflow.insert_step(2, "C")
        self.assertEqual(len(self.workflow.steps), 3)
        for i, index in enumerate(indices):
            self.workflow.remove_step(index)
        self.assertEqual(len(self.workflow.steps), len(expected_labels))
        ordered_labels = [s.label for s in self.workflow.steps]
        self.assertListEqual(ordered_labels, expected_labels)

    def test_remove_error(self):
        self.assertEqual(len(self.workflow.steps), 0)
        self.assertRaises(IndexError, self.workflow.remove_step, 0)

        self.assertRaises(IndexError, self.workflow.remove_step, 10)

        self.workflow.insert_step(0, "A")
        self.workflow.remove_step(0)
        self.assertRaises(IndexError, self.workflow.remove_step, 0)

        self.workflow.insert_step(0, "A")
        self.workflow.remove_step(0)
        self.assertRaises(TypeError, self.workflow.remove_step, None)

    # def test_reorder(self, current_index, new_index):
    #     pass
    #
    # def test_rename(self, index, new_name):
    #     pass
    #
    # def test_reset(self):
    #     pass


# class TestDefaultStep(unittest.TestCase):
#     @parameterized.expand([
#
#     ])

#     def test_insert(self, index, name):
#         branch_blocks = workflow.secondary_branch_blocks(block)
#         self.assertEqual(len(branch_blocks), len(expected_branch_blocks))
#         for branch_block in expected_branch_blocks:
#             self.assertIn(branch_block, branch_blocks)