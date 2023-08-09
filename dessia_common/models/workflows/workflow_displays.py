""" A workflow that has 2D, 3D and MD displays. """

from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import ClassMethod, PlotData, CadView, Markdown
from dessia_common.forms import StandaloneObject
from dessia_common.typings import ClassMethodType, CadViewType, PlotDataType, MarkdownType

cadview_selector = CadViewType(class_=StandaloneObject, name="some_cad_selector")
cadview = CadView(selector=cadview_selector, name="3D")

plotdata_selector = PlotDataType(class_=StandaloneObject, name="Scatter Plot")
plotdata = PlotData(selector=plotdata_selector, name="2D", load_by_default=True)

markdown_selector = MarkdownType(class_=StandaloneObject, name="My Markdown Selector")
markdown = Markdown(selector=markdown_selector, name="MD", load_by_default=False)

cmt = ClassMethodType(class_=StandaloneObject, name="generate")
cmb = ClassMethod(method_type=cmt, name="Generator")


plotdatatest = PlotData(name="2DTest", selector="2DTest")
markdowntest = Markdown(name="MDTest", selector="MDTest")


blocks = [cmb, cadview, plotdata, markdown, plotdatatest, markdowntest]
pipes = [Pipe(input_variable=cmb.outputs[0], output_variable=cadview.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=plotdata.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=markdown.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=plotdatatest.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=markdowntest.inputs[0])]

workflow = Workflow(blocks=blocks, pipes=pipes, output=cmb.outputs[0])
