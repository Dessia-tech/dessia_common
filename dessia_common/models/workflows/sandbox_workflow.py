from dessia_common.forms import Generator, StandaloneObject
from dessia_common.workflow.blocks import (InstantiateModel, ModelMethod, GetModelAttribute, MultiPlot, Unpacker,
                                           CadView, Markdown, PlotData, Export, ConcatenateStrings, Archive,
                                           ClassMethod, SetModelAttribute, Sequence, WorkflowBlock, ForEach)
from dessia_common.typings import MethodType, AttributeType, CadViewType, MarkdownType, PlotDataType, ClassMethodType
from dessia_common.workflow.core import Workflow, Variable, Pipe

documentation = """> test

 some ~~markdown~~ _feature_ while writing **dummy** `workflow` doc"""

generator_block = InstantiateModel(model_class=Generator, name="Generator")
generate_block = ModelMethod(method_type=MethodType(Generator, 'generate'), name="Generate")
models_block = GetModelAttribute(attribute_type=AttributeType(Generator, name="models"), name="Models")
mp_block = MultiPlot(selector_name='Multiplot',
                     attributes=['standalone_subobject/intarg', 'standalone_subobject/floatarg'],
                     name="MP", load_by_default=True)
unpacker_block = Unpacker(indices=[0], name="Unpacker")
cad_block = CadView(selector=CadViewType(class_=StandaloneObject, name='CAD With Selector'), name="CAD",
                    load_by_default=False)
md_block = Markdown(selector=MarkdownType(class_=StandaloneObject, name='Markdown'), name="MD", load_by_default=False)
pd_block = PlotData(selector=PlotDataType(class_=StandaloneObject, name='Scatter Plot'), name="PD",
                    load_by_default=False)
json_block = Export(method_type=MethodType(StandaloneObject, 'save_to_stream'), filename='export_json',
                    extension='json', text=True, name="JSON")
json_block.inputs[1].lock()
xlsx_block = Export(method_type=MethodType(Generator, 'to_xlsx_stream'), filename='export', extension='xlsx',
                    text=False, name="Export")
concat_block = ConcatenateStrings(number_elements=2, separator='_', name='Concat')
zip_block = Archive(number_exports=2, filename='archive', name="ZIP")
import_block = ClassMethod(method_type=ClassMethodType(StandaloneObject, 'generate_from_text'), name="From File")
setattr_block = SetModelAttribute(attribute_type=AttributeType(StandaloneObject, name="tuple_arg"), name="Set Tuple")
sequence_block = Sequence(number_arguments=2, name="Sequence")

# --- Subworkflow ---
documentation = """"""

sub_block_0 = ModelMethod(method_type=MethodType(StandaloneObject, 'count_until'), name="Count")
sub_blocks = [sub_block_0]


sub_pipes = []

sub_workflow = Workflow(blocks=sub_blocks, pipes=sub_pipes, output=sub_block_0.outputs[1], documentation=documentation,
                        name="Packed Workflow")
# --- End Subworkflow ---

wfblock = WorkflowBlock(workflow=sub_workflow, name="")
foreach_block = ForEach(workflow_block=wfblock, iter_input_index=0, name="")
blocks = [generator_block, generate_block, models_block, mp_block, unpacker_block, cad_block, md_block, pd_block,
          json_block, xlsx_block, concat_block, zip_block, import_block, setattr_block, sequence_block, foreach_block]

variable_0 = Variable(name='_result_name_', label='Result Name', type_=str)
variable_1 = Variable(name='Substring 2', label='XLSX Filename Suffix', type_=str)
variable_1.lock("My Locked NBV")

pipe_0 = Pipe(variable_0, generator_block.inputs[3])
pipe_1 = Pipe(generator_block.outputs[0], generate_block.inputs[0])
pipe_2 = Pipe(generate_block.outputs[1], models_block.inputs[0])
pipe_3 = Pipe(generate_block.outputs[0], mp_block.inputs[0])
pipe_4 = Pipe(models_block.outputs[0], unpacker_block.inputs[0])
pipe_5 = Pipe(unpacker_block.outputs[0], cad_block.inputs[0])
pipe_6 = Pipe(unpacker_block.outputs[0], md_block.inputs[0])
pipe_7 = Pipe(unpacker_block.outputs[0], pd_block.inputs[0])
pipe_8 = Pipe(unpacker_block.outputs[0], json_block.inputs[0])
pipe_9 = Pipe(concat_block.outputs[0], xlsx_block.inputs[1])
pipe_10 = Pipe(variable_1, concat_block.inputs[1])
pipe_11 = Pipe(variable_0, concat_block.inputs[0])
pipe_12 = Pipe(generator_block.outputs[0], xlsx_block.inputs[0])
pipe_13 = Pipe(json_block.outputs[0], zip_block.inputs[0])
pipe_14 = Pipe(xlsx_block.outputs[0], zip_block.inputs[1])
pipe_15 = Pipe(import_block.outputs[0], setattr_block.inputs[0])
pipe_16 = Pipe(setattr_block.outputs[0], sequence_block.inputs[0])
pipe_17 = Pipe(unpacker_block.outputs[0], sequence_block.inputs[1])
pipe_18 = Pipe(sequence_block.outputs[0], foreach_block.inputs[0])
pipes = [pipe_0, pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12,
         pipe_13, pipe_14, pipe_15, pipe_16, pipe_17, pipe_18]

workflow = Workflow(blocks=blocks, pipes=pipes, output=generate_block.outputs[1], documentation=documentation,
                    name="Generator 2024.02")


workflow.insert_step(None, "A")
workflow.insert_step(1, "B")
workflow.insert_step(1, "C")

for i, input_ in enumerate(workflow.inputs):
    step = workflow._steps[i % len(workflow._steps)]
    workflow.change_input_step(input_=input_, step=step)

# workflow.log_steps("Before Setting Display")
#
output = unpacker_block.outputs[0]
workflow.add_step_display(output, workflow.steps[0], StandaloneObject.display_settings()[0])

# workflow.remove_step(workflow.steps[1])

# workflow.log_steps("After Setting Display")

# for input_ in reversed(workflow.inputs):
#     workflow.change_input_step(input_=input_, step=workflow.steps[0])
#
# workflow.log_steps("C")

