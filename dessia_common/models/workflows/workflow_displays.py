from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import ClassMethod, PlotData, CadView, Markdown
from dessia_common.typings import ClassMethodType
from dessia_common.forms import StandaloneObject

cmt = ClassMethodType(class_=StandaloneObject, name="generate")
cmb = ClassMethod(method_type=cmt, name="Generator")

cadview = CadView("3D")
plotdata = PlotData("2D")
markdown = Markdown("MD")

blocks = [cmb, cadview, plotdata, markdown]
pipes = [Pipe(input_variable=cmb.outputs[0], output_variable=cadview.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=plotdata.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=markdown.inputs[0])]

workflow = Workflow(blocks=blocks, pipes=pipes, output=cmb.outputs[0])
