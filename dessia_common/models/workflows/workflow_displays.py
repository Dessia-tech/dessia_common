""" A workflow that has 2D, 3D and MD displays. """

from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import ClassMethod, PlotData, Markdown
from dessia_common.forms import StandaloneObject
from dessia_common.typings import ClassMethodType, PlotDataType, MarkdownType

plotdata_selector = PlotDataType(class_=StandaloneObject, name="Scatter Plot")
plotdata = PlotData(selector=plotdata_selector, name="2D", load_by_default=True)

markdown_selector = MarkdownType(class_=StandaloneObject, name="Markdown")
markdown = Markdown(selector=markdown_selector, name="MD", load_by_default=False)

cmt = ClassMethodType(class_=StandaloneObject, name="generate")
cmb = ClassMethod(method_type=cmt, name="Generator")


blocks = [cmb, plotdata, markdown]
pipes = [Pipe(input_variable=cmb.outputs[0], output_variable=plotdata.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=markdown.inputs[0])]

workflow = Workflow(blocks=blocks, pipes=pipes, output=cmb.outputs[0])
