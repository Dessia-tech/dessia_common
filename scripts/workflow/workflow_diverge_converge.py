from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import Unpacker, Sequence, InstantiateModel, ModelMethod
from dessia_common.typings import MethodType

from dessia_common.forms import Generator

generate = InstantiateModel(model_class=Generator, name="Test1")
method_type = MethodType(class_=Generator, name="generate")
method = ModelMethod(method_type=method_type)

unpack1 = Unpacker(indices=[0, 2])
unpack2 = Unpacker(indices=[1, 3])

sequence = Sequence(number_arguments=4)

blocks = [generate, method, unpack1, unpack2, sequence]
pipes = [Pipe(input_variable=generate.outputs[0], output_variable=method.inputs[0]),
         Pipe(input_variable=method.outputs[0], output_variable=unpack1.inputs[0]),
         Pipe(input_variable=method.outputs[0], output_variable=unpack2.inputs[0]),
         Pipe(input_variable=unpack1.outputs[0], output_variable=sequence.inputs[0]),
         Pipe(input_variable=unpack1.outputs[1], output_variable=sequence.inputs[2]),
         Pipe(input_variable=unpack2.outputs[0], output_variable=sequence.inputs[1]),
         Pipe(input_variable=unpack2.outputs[1], output_variable=sequence.inputs[3])]

workflow_ = Workflow(blocks=blocks, pipes=pipes, output=sequence.outputs[0])

assert len(workflow_.blocks) == len(workflow_.runtime_blocks) and set(workflow_.runtime_blocks) == set(workflow_.blocks)
