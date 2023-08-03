""" A workflow that has 2D, 3D and MD displays. """

from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import ClassMethod, PlotData, CadView, Markdown, CadViewFromDecorator
from dessia_common.typings import ClassMethodType
from dessia_common.forms import StandaloneObject

cmt = ClassMethodType(class_=StandaloneObject, name="generate")
cmb = ClassMethod(method_type=cmt, name="Generator")

cadview = CadViewFromDecorator("3D")
plotdata = PlotData("2D")
markdown = Markdown("MD")
plotdatatest = PlotData(name="2DTest", selector="2DTest", type_="plot_data_test")
markdowntest = Markdown(name="MDTest", selector="MDTest", type_="markdown_test")


blocks = [cmb, cadview, plotdata, markdown, plotdatatest, markdowntest]
pipes = [Pipe(input_variable=cmb.outputs[0], output_variable=cadview.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=plotdata.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=markdown.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=plotdatatest.inputs[0]),
         Pipe(input_variable=cmb.outputs[0], output_variable=markdowntest.inputs[0])]

workflow = Workflow(blocks=blocks, pipes=pipes, output=cmb.outputs[0])
