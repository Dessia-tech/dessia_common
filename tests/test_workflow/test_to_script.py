import unittest
from typing import List, Dict, Tuple, Union

from dessia_common.forms import EmbeddedSubobject, StandaloneBuiltinsSubobject
from dessia_common.measures import Measure, Distance
from parameterized import parameterized

from dessia_common.typings import MethodType, MarkdownType, PlotDataType, InstanceOf
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
    def __init__(self, name: str = "", connections: List[ConnectionNode] = None,
                 binary_files: Tuple[BinaryFile, BinaryFile] = None,
                 integers: List[int] = None, floats: List[float] = None, strings: List[str] = None,
                 dictionary: Dict[str, int] = None,
                 nested_tuples: List[List[Tuple[float, float]]] = None,
                 embedded_object: InstanceOf[EmbeddedSubobject] = None,
                 union_object: Union[EmbeddedSubobject, StandaloneBuiltinsSubobject] = None,
                 nested_lists: List[List[List[Tuple[float, float]]]] = None,
                 measure_object: Measure = None,
                 distance_object: Distance = None):
        self.connections = connections
        self.binary_files = binary_files
        self.integers = integers
        self.floats = floats
        self.strings = strings
        self.dictionary = dictionary
        self.nested_tuples = nested_tuples
        self.embedded_object = embedded_object
        self.union_object = union_object
        self.nested_lists = nested_lists
        self.measure_object = measure_object
        self.distance_object = distance_object
        super().__init__(name=name)

    def run_driver(self):
        pass

    @markdown_view("Markdown")
    def to_markdown(self):
        return super().to_markdown(self=self)

    @plot_data_view(selector="PlotData", load_by_default=True)
    def plot_data(self, reference_path: str = "#", **kwargs):
        return []


# Blocks
block_0 = [InstantiateModel(model_class=ExperimentObject, name="MyObjInstance", position=[-1910.252, 116.630]),
           'InstantiateModel(model_class=ExperimentObject, name="MyObjInstance", position=[-1910.252, 116.63])']

block_1 = [ModelMethod(method_type=MethodType(ExperimentObject, 'run_driver'), name="run_driver_custom",
                       position=[-13.917, 49.856]),
           'ModelMethod(method_type=MethodType(ExperimentObject, \'run_driver\'), '
           'name="run_driver_custom", position=[-13.917, 49.856])']

block_2 = [
    Markdown(selector=MarkdownType(class_=ExperimentObject, name="Markdown"), name='Display markdown',
             load_by_default=False, position=(0, 0)),
    'Markdown(selector=MarkdownType(class_=ExperimentObject, name=\'Markdown\'), name="Display markdown", '
    'load_by_default=False, position=(0, 0))']
block_4 = [PlotData(selector=PlotDataType(class_=ExperimentObject, name="PlotData"), name='Display plot_data',
                    load_by_default=True, position=(0, 0)), 'PlotData(selector=PlotDataType(class_=ExperimentObject, '
                                                            'name=\'PlotData\'), name="Display plot_data", '
                                                            'load_by_default=True, position=(0, 0))']

block_5 = [Export(method_type=MethodType(ExperimentObject, 'save_to_stream'), name='Export', filename="filename",
                  extension="json", text=True, position=(99, 88.88)),
           'Export(method_type=MethodType(ExperimentObject, \'save_to_stream\'),'
           ' filename=\'filename\', extension=\'json\', text=True,'
           ' name="Export", position=(99, 88.88))']

blocks = [block_0, block_1, block_2, block_4, block_5]

# Variables
variable_list = [
    [
        Variable(name='variable_0', position=[-6.77, 34.65], type_=str),
        "Variable(name='variable_0', position=[-6.77, 34.65], type_=str)"
    ],
    [
        Variable(name='variable_1', position=[-632.77, 34.65], type_=List[BinaryFile]),
        "Variable(name='variable_1', position=[-632.77, 34.65], type_=List[BinaryFile])"
    ],
    [
        Variable(name='variable_2', position=[-2342.61, 18.72], type_=List[ConnectionNode]),
        "Variable(name='variable_2', position=[-2342.61, 18.72], type_=List[ConnectionNode])"
    ],
    [
        Variable(name='variable_3', position=[197.91, 95.02], type_=List[int]),
        "Variable(name='variable_3', position=[197.91, 95.02], type_=List[int])"
    ],
    [
        Variable(name='variable_4', position=[-1035.71, -269.35], type_=List[str]),
        "Variable(name='variable_4', position=[-1035.71, -269.35], type_=List[str])"
    ],
    [
        Variable(name='variable_5', position=[2620.96, -486.85], type_=List[bool]),
        "Variable(name='variable_5', position=[2620.96, -486.85], type_=List[bool])"
    ],
    [
        Variable(name='variable_6', position=[2620.96, -486.85], type_=Dict[str, int]),
        "Variable(name='variable_6', position=[2620.96, -486.85], type_=Dict[str, int])"
    ],
    [
        Variable(name='variable_7', position=[2620.9, -486.8], type_=List[List[Tuple[float, float]]]),
        "Variable(name='variable_7', position=[2620.9, -486.8], type_=List[List[Tuple[float, float]]])"
    ],
    [
        Variable(name='variable_8', position=[2620.96, -486.84], type_=InstanceOf[EmbeddedSubobject]),
        "Variable(name='variable_8', position=[2620.96, -486.84], type_=InstanceOf[EmbeddedSubobject])"
    ],
    [
        Variable(name='variable_9', position=[2.6, -4.8], type_=Union[EmbeddedSubobject, StandaloneBuiltinsSubobject]),
        "Variable(name='variable_9', position=[2.6, -4.8], type_=Union[EmbeddedSubobject, StandaloneBuiltinsSubobject])"
    ],
    [
        Variable(name='variable_10', position=[2620.96, -486.84], type_=List[List[List[Tuple[float, float]]]]),
        "Variable(name='variable_10', position=[2620.96, -486.84], type_=List[List[List[Tuple[float, float]]]])"
    ],
    [
        Variable(name='variable_11', position=[26.96, -486.84], type_=Measure),
        "Variable(name='variable_11', position=[26.96, -486.84], type_=Measure)"
    ],
    [
        Variable(name='variable_12', position=[26.96, -486.84], type_=Distance),
        "Variable(name='variable_12', position=[26.96, -486.84], type_=Distance)"
    ]
]

# Pipes
pipe_1 = Pipe(blocks[0][0].outputs[0], blocks[1][0].inputs[0])
pipe_2 = Pipe(variable_list[1][0], blocks[0][0].inputs[2])
pipe_3 = Pipe(variable_list[2][0], blocks[0][0].inputs[1])
pipe_4 = Pipe(variable_list[3][0], blocks[0][0].inputs[3])
pipe_5 = Pipe(variable_list[4][0], blocks[0][0].inputs[5])
pipe_6 = Pipe(variable_list[5][0], blocks[0][0].inputs[4])
pipe_7 = Pipe(variable_list[0][0], blocks[0][0].inputs[0])
pipe_8 = Pipe(variable_list[6][0], blocks[0][0].inputs[6])
pipe_9 = Pipe(variable_list[7][0], blocks[0][0].inputs[7])

pipe_10 = Pipe(variable_list[8][0], blocks[0][0].inputs[8])
pipe_11 = Pipe(variable_list[9][0], blocks[0][0].inputs[9])
pipe_12 = Pipe(variable_list[10][0], blocks[0][0].inputs[10])
pipe_13 = Pipe(variable_list[11][0], blocks[0][0].inputs[11])
pipe_14 = Pipe(variable_list[12][0], blocks[0][0].inputs[12])

pipe_15 = Pipe(blocks[1][0].outputs[0], blocks[2][0].inputs[0])
pipe_16 = Pipe(blocks[1][0].outputs[0], blocks[3][0].inputs[0])

workflow = Workflow([block[0] for block in blocks],
                    [
                        pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7,
                        pipe_8, pipe_9, pipe_10, pipe_11, pipe_12, pipe_13, pipe_14, pipe_15, pipe_16
                    ],
                    output=blocks[1][0].outputs[0], name="TestExport")

workflow_script = workflow.to_script()


class WorkflowToScriptTest(unittest.TestCase):

    @parameterized.expand(blocks)
    def test_blocks(self, block, expected_declaration):
        self.assertEqual(block._to_script("").declaration, expected_declaration)

    @parameterized.expand(variable_list)
    def test_variables(self, variable, expected_declaration):
        self.assertEqual(variable._to_script().declaration, expected_declaration)

    @parameterized.expand(variable_list)
    def test_workflow_variables(self, _, expected_declaration):
        self.assertIn(expected_declaration, workflow_script)

    @parameterized.expand(blocks)
    def test_workflow_blocks(self, _, expected_declaration):
        self.assertIn(expected_declaration, workflow_script)

