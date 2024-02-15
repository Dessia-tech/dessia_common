import unittest
from typing import List, Dict, Tuple
from dessia_common.measures import Measure, Distance
from parameterized import parameterized

from dessia_common.typings import MethodType, MarkdownType, PlotDataType
from dessia_common.workflow.core import Pipe, Workflow, Variable
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod, PlotData, Markdown, Export
from dessia_common.decorators import markdown_view, plot_data_view
from dessia_common.files import BinaryFile
from dessia_common.core import DessiaObject


class ConnectionNode(DessiaObject):
    def __init__(self, name: str = "", number_connection: int = 2, file: BinaryFile = None):
        self.number_connection = number_connection
        self.file = file
        super().__init__(name=name)


class ExperimentObject(DessiaObject):
    def __init__(self, name: str = "",
                 connections: List[ConnectionNode] = None,
                 binary_files: Tuple[BinaryFile, BinaryFile] = None,
                 integers: List[int] = None,
                 floats: List[float] = None,
                 strings: List[str] = None,
                 dictionary: Dict[str, int] = None,
                 nested_tuples: List[List[Tuple[float, float]]] = None,
                 nested_lists: List[List[List[Tuple[float, float]]]] = None,
                 measure_object: Measure = None,
                 distance_object: Distance = None):
        if nested_tuples is None:
            nested_tuples = [[(1, 2), (1, 4)], [(0, 2), (3, 9)]]
        self.connections = connections
        self.binary_files = binary_files
        self.integers = integers
        self.floats = floats
        self.strings = strings
        self.dictionary = dictionary
        self.nested_tuples = nested_tuples
        self.nested_lists = nested_lists
        self.measure_object = measure_object
        self.distance_object = distance_object
        super().__init__(name=name)

    def run_driver(self):
        return {"A": 0.1, "B": 0.1}

    @markdown_view("MarkdownExperimentObject")
    def to_markdown(self):
        return super().to_markdown()

    @plot_data_view(selector="PlotDataExperimentObject", load_by_default=True)
    def plot_data(self, reference_path: str = "#", **kwargs):
        return []


class Result(DessiaObject):
    def __init__(self, data: Dict[str, float], name: str = ""):
        self.data = data

        DessiaObject.__init__(self, name=name)

    @markdown_view("MarkdownResult")
    def to_markdown(self):
        return super().to_markdown()

    @plot_data_view(selector="PlotDataResult", load_by_default=True)
    def plot_data(self, reference_path: str = "#", **kwargs):
        return []


block_0 = InstantiateModel(model_class=ExperimentObject, name="MyObjInstance", position=[-1910.252, 116.630])

block_1 = ModelMethod(method_type=MethodType(ExperimentObject, 'run_driver'), name="run_driver_custom",
                      position=[-13.917, 49.856])

block_3 = InstantiateModel(model_class=Result, name="Result", position=[-19.252, 16.63])

block_2 = Markdown(selector=MarkdownType(class_=ExperimentObject, name="Markdown"), name='Display markdown',
                   load_by_default=False, position=(0, 0))
block_4 = PlotData(selector=PlotDataType(class_=ExperimentObject, name="PlotData"), name='Display plot_data',
                   load_by_default=True, position=(0, 0))

block_5 = Export(method_type=MethodType(ExperimentObject, 'save_to_stream'), name='Export', filename="filename",
                 extension="json", text=True, position=(99, 88.88))

variable_0 = Variable(name='variable_0', position=[-6.7679391946565, 34.65331017231682], type_=str)
variable_1 = Variable(name='variable_1', position=[-2342.6107573347076, 18.721713795811482], type_=List[ConnectionNode])
variable_2 = Variable(name='variable_2', position=[-632.7679391946565, 34.65331017231682], type_=List[BinaryFile])
variable_3 = Variable(name='variable_3', position=[197.91117106810384, 95.02100327832922], type_=List[int])

pipe_0 = Pipe(variable_0, block_0.inputs[0])
pipe_1 = Pipe(variable_1, block_0.inputs[1])
pipe_2 = Pipe(variable_2, block_0.inputs[2])
pipe_3 = Pipe(variable_3, block_0.inputs[3])

pipe_4 = Pipe(block_1.outputs[1], block_3.inputs[0])

pipe_5 = Pipe(block_1.outputs[0], block_2.inputs[0])
pipe_6 = Pipe(block_1.outputs[0], block_4.inputs[0])
pipe_7 = Pipe(block_0.outputs[0], block_5.inputs[0])
pipe_8 = Pipe(block_0.outputs[0], block_1.inputs[0])

# Workflow
workflow = Workflow(blocks=[block_0,
                            block_1,
                            block_2,
                            block_3,
                            block_4,
                            block_5
                            ],
                    pipes=[
                        # pipe_0,
                        pipe_1,
                        pipe_2,
                        pipe_3,
                        pipe_4,
                        pipe_5,
                        pipe_6,
                        pipe_7,
                        pipe_8
                    ],
                    output=block_0.outputs[0])

# WorkflowRun
strings = ["ABCD", "EFG", "HIJ"]
floats = [0.1, 0.4, 100.1]
dict_val = {"A": 2, "B": 4}
bin_file = [BinaryFile(), BinaryFile()]
my_objs = [ConnectionNode(number_connection=2), ConnectionNode(number_connection=1),
           ConnectionNode(number_connection=3)]
numbers = [1, 2, 3, 4]
list_tuple = [[(1, 2), (1, 4)], [(0, 2), (3, 9)]]

input_values = {
    workflow.input_index(block_0.inputs[0]): "test",
    workflow.input_index(variable_1): my_objs,
    workflow.input_index(variable_2): bin_file,
    workflow.input_index(variable_3): numbers,
    workflow.input_index(block_0.inputs[4]): floats,
    workflow.input_index(block_0.inputs[5]): strings,
    workflow.input_index(block_0.inputs[6]): dict_val,
    workflow.input_index(block_0.inputs[7]): list_tuple,
}
workflow_run = workflow.run(input_values=input_values)
script_ = workflow_run.to_script().declaration

variables = [
    "value_0_0 = 'test'",
    "value_0_4 = [0.1, 0.4, 100.1]",
    "value_0_5 = ['ABCD', 'EFG', 'HIJ']",
    "value_0_6 = {'A': 2, 'B': 4}",
    "value_0_7 = [[(1, 2), (1, 4)], [(0, 2), (3, 9)]]",
    "value_0_8 = None",
    "value_0_9 = None",
    "value_0_10 = None",
    "value_3_1 = ''",
    "value_5_1 = 'filename'",
    "value_0_ = [\n\tConnectionNode('Set your arguments here'),\n\tConnectionNode('Set your arguments here'),"
    "\n\tConnectionNode('Set your arguments here')\n]",
    "value_1_ = [\n\tBinaryFile.from_file('Set your filepath here'),\n\tBinaryFile.from_file('Set your filepath "
    "here')\n]",
    "value_2_ = [1, 2, 3, 4]",
]

input_values = [
    "workflow.input_index(block_0.inputs[0]): value_0_0,",
    "workflow.input_index(block_0.inputs[4]): value_0_4,",
    "workflow.input_index(block_0.inputs[5]): value_0_5,",
    "workflow.input_index(block_0.inputs[6]): value_0_6,",
    "workflow.input_index(block_0.inputs[7]): value_0_7,",
    "workflow.input_index(block_0.inputs[8]): value_0_8,",
    "workflow.input_index(block_0.inputs[9]): value_0_9,",
    "workflow.input_index(block_0.inputs[10]): value_0_10,",
    "workflow.input_index(block_3.inputs[1]): value_3_1,",
    "workflow.input_index(block_5.inputs[1]): value_5_1,",
    "workflow.input_index(variable_0): value_0_,",
    "workflow.input_index(variable_1): value_1_,",
    "workflow.input_index(variable_2): value_2_,",

]


class TestWorkflowScript(unittest.TestCase):

    @parameterized.expand(variables)
    def test_variables_in_script(self, variable):
        self.assertIn(variable, script_)

    @parameterized.expand(input_values)
    def test_input_values_in_script(self, input_value):
        self.assertIn(input_value, script_)

    def test_workflow_run_in_script(self):
        self.assertIn("workflow_run = workflow.run(input_values=input_values)", script_)


if __name__ == '__main__':
    unittest.main()
