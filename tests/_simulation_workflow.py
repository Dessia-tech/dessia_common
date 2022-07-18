import dessia_common.workflow as dcw
import dessia_common.workflow.blocks as dcw_blocks
import dessia_common.typings as dct

import dessia_common.tests

block_0 = dcw_blocks.InstantiateModel(dessia_common.tests.System, name='Instantiate Generator')
block_1 = dcw_blocks.ModelMethod(method_type=dct.MethodType(dessia_common.tests.System, 'power_simulation'), name='Generator Generate')
blocks = [block_0, block_1]
pipe_0 = dcw.Pipe(block_0.outputs[0]
, block_1.inputs[0]
)
pipes = [pipe_0]
workflow = dcw.Workflow(blocks, pipes, output=block_1.outputs[0],name='')
