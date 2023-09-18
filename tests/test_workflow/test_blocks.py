import unittest
from parameterized import parameterized
from dessia_common.workflow.blocks import (InstantiateModel, ClassMethod, ModelMethod, Sequence, Concatenate,
                                           WorkflowBlock, ForEach, Unpacker, Flatten, Product, Filter, CadView,
                                           Markdown, PlotData, GetModelAttribute, SetModelAttribute, Sum, Substraction,
                                           ConcatenateStrings, Export, Archive)


class TestBlocks(unittest.TestCase):

    @parameterized.expand([
        (InstantiateModel,),
        (ClassMethod,),
        (ModelMethod,),
        (Sequence,),
        (Concatenate,),
        (WorkflowBlock,),
        (ForEach,),
        (Unpacker,),
        (Flatten,),
        (Product,),
        (Filter,),
        (CadView,),
        (Markdown,),
        (PlotData,),
        (GetModelAttribute,),
        (SetModelAttribute,),
        (Sum,),
        (Substraction,),
        (ConcatenateStrings,),
        (Export,),
        (Archive,)
     ])
    def test_block_schemas(self, block):
        block.schema()
