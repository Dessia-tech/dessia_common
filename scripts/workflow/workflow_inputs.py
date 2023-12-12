from dessia_common.forms import Generator, StandaloneObject
from dessia_common.workflow.blocks import InstantiateModel, ClassMethod, Sequence
from dessia_common.typings import ClassMethodType
from dessia_common.workflow.core import TypedVariable, Pipe, Workflow
from dessia_common.files import StringFile

block_0 = InstantiateModel(model_class=Generator, name='Simple Inputs',
                           position=(41.40861618798957, -84.52480417754569))
block_1 = ClassMethod(method_type=ClassMethodType(StandaloneObject, 'generate_from_bin'), name='Binary Input',
                      position=(42.59477666877444, 94.63608398070332))
block_2 = ClassMethod(method_type=ClassMethodType(StandaloneObject, 'generate_from_text'), name='Text Input',
                      position=(42.32922270510359, 184.5889177760892))
block_3 = InstantiateModel(model_class=StandaloneObject, name='Custom & Complex Inputs',
                           position=(39.02562760496994, 274.3967730534379))
block_4 = Sequence(number_arguments=4, name='Packer', position=(482.55623557963327, 33.78454658062492))
blocks = [block_0, block_1, block_2, block_3, block_4]

variable_0 = TypedVariable(name='Result Name', position=[-353.5235486758587, 234.9165175396766], type_=str)
variable_1 = TypedVariable(name='stream', position=[-131.55327139280053, 248.81817426839467], type_=StringFile)

pipe_0 = Pipe(variable_0, block_3.inputs[12])
pipe_1 = Pipe(variable_0, block_0.inputs[3])
pipe_2 = Pipe(variable_1, block_2.inputs[0])
pipe_3 = Pipe(block_0.outputs[0], block_4.inputs[0])
pipe_4 = Pipe(block_1.outputs[0], block_4.inputs[1])
pipe_5 = Pipe(block_2.outputs[0], block_4.inputs[2])
pipe_6 = Pipe(block_3.outputs[0], block_4.inputs[3])
pipes = [pipe_0, pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6]

workflow = Workflow(blocks, pipes, output=block_4.outputs[0], name='Test Inputs')

assert workflow.has_file_inputs
assert len(workflow.file_inputs) == 2

file_input_indices = [workflow.input_index(i) for i in workflow.file_inputs]

assert file_input_indices == [3, 17]
assert workflow.is_variable_nbv(workflow.file_inputs[1])


