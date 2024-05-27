import unittest
from parameterized import parameterized
from dessia_common.workflow import Workflow
from dessia_common.workflow.blocks import (InstantiateModel, ClassMethod, ModelMethod, Sequence, Concatenate,
                                           Unpacker, Flatten, Product, Filter, CadView, Markdown, PlotData,
                                           GetModelAttribute, SetModelAttribute, Sum, Substraction, ConcatenateStrings,
                                           Export, Archive, MultiPlot)
from dessia_common.forms import StandaloneObject, Generator
from dessia_common.typings import (MethodType, ClassMethodType, AttributeType, CadViewType, MarkdownType, PlotDataType)


class TestBlocks(unittest.TestCase):
    @parameterized.expand([
        (InstantiateModel(StandaloneObject),),
        (ClassMethod(ClassMethodType(class_=StandaloneObject, name="generate")),),
        (ModelMethod(ClassMethodType(class_=Generator, name="generate")),),
        (ModelMethod(
            method_type=ClassMethodType(class_=StandaloneObject, name="count_until"),
            name="ModelMethod with args"
        ),),
        (Sequence(2),),
        (Concatenate(),),
        # (WorkflowBlock,),
        # (ForEach,),
        (Unpacker([0]),),
        (Flatten(),),
        (Product(2),),
        (Filter([]),),
        (CadView(CadViewType(class_=StandaloneObject, name="CAD With Selector")),),
        (Markdown(MarkdownType(class_=StandaloneObject, name="Markdown")),),
        (PlotData(PlotDataType(class_=StandaloneObject, name="2D View")),),
        (MultiPlot(selector_name="Multiplot", attributes=[]),),
        (GetModelAttribute(AttributeType(class_=StandaloneObject, name="models")),),
        (SetModelAttribute(AttributeType(class_=StandaloneObject, name="standalone_subobject/intarg")),),
        (Sum(),),
        (Substraction(),),
        (ConcatenateStrings(),),
        (Export(method_type=MethodType(class_=StandaloneObject, name="save_to_stream"), text=True, extension="json"),),
        (Export(method_type=MethodType(class_=StandaloneObject, name="to_xlsx_stream"), text=False, extension="xlsx"),),
        (Archive(),)
    ])
    def test_display_blocks(self, block):
        dict_ = Workflow(blocks=[block], pipes=[], output=block.outputs[0], name=f"Workflow - {block.name}").to_dict()
        workflow = Workflow.dict_to_object(dict_)
        for workflow_input, block_input in zip(workflow.inputs, workflow.blocks[0].inputs):
            self.assertIs(workflow_input, block_input)



