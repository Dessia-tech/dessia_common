import unittest
from parameterized import parameterized
from dessia_common.models.workflows.sandbox_workflow import (workflow, generator_block, generate_block, zip_block,
                                                             foreach_block, cad_block, models_block, unpacker_block,
                                                             concat_block, json_block, xlsx_block, sequence_block,
                                                             setattr_block, import_block)


class TestBlocksBranches(unittest.TestCase):
    @parameterized.expand([
        (cad_block, [models_block, unpacker_block, cad_block]),
        (zip_block, [models_block, unpacker_block, concat_block, json_block, xlsx_block, zip_block]),
        (foreach_block, [models_block, unpacker_block, sequence_block, import_block, setattr_block, foreach_block]),
        (generator_block, [generator_block]),  # Should it be an error or None ? (block not a secondary block)
        (generate_block, [generate_block])  # Should it be an error or None ? (block not a secondary block)
    ])
    def test_secondary_branches(self, block, expected_branch_blocks):
        branch_blocks = workflow.secondary_branch_blocks(block)
        self.assertEqual(len(branch_blocks), len(expected_branch_blocks))
        for branch_block in expected_branch_blocks:
            self.assertIn(branch_block, branch_blocks)

    @parameterized.expand([
        (cad_block, [models_block, unpacker_block, cad_block, generator_block, generate_block]),
        (zip_block, [
            models_block, unpacker_block, concat_block, json_block, xlsx_block, zip_block,
            generator_block, generate_block
        ]),
        (foreach_block, [
            models_block, unpacker_block, sequence_block, import_block, setattr_block, foreach_block,
            generator_block, generate_block
        ]),
        (generator_block, [generator_block]),
        (generate_block, [generator_block, generate_block])
    ])
    def test_full_branches(self, block, expected_branch_blocks):
        branch_blocks = workflow.upstream_branch(block)
        self.assertEqual(len(branch_blocks), len(expected_branch_blocks))
        for branch_block in expected_branch_blocks:
            self.assertIn(branch_block, branch_blocks)
