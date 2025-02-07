from dessia_common.tests import System
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod
from dessia_common.typings import MethodType
from dessia_common.workflow.core import Pipe, Workflow

documentation = """"""

block_0 = InstantiateModel(model_class=System, name="Instantiate Generator", position=(0, 0))
block_1 = ModelMethod(method_type=MethodType(System, 'power_simulation'), name="Generator Generate", position=(0, 0))
blocks = [block_0, block_1]


pipe_0 = Pipe(block_0.outputs[0], block_1.inputs[0])
pipes = [pipe_0]

workflow = Workflow(blocks, pipes, output=block_1.outputs[0], documentation=documentation, name="")
