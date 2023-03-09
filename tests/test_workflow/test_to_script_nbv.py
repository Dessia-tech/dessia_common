import unittest
from typing import Dict, List

from dessia_common.files import StringFile
from dessia_common.forms import (EmbeddedSubobject,
                                 StandaloneBuiltinsSubobject, StandaloneObject)
from dessia_common.tests import Car
from dessia_common.typings import ClassMethodType, InstanceOf
from dessia_common.workflow.blocks import ClassMethod, InstantiateModel
from dessia_common.workflow.core import Pipe, TypedVariable, Workflow


class WorkflowToScriptTest(unittest.TestCase):

    def test_simple_equality(self):
        class_method = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='car_from_csv',
                              position=[-27.154996774790852, -138.67410502990185])
        instantiate_object = InstantiateModel(model_class=StandaloneObject, name='',
                                   position=[-6.746717471743784, 60.49659346366095])
        blocks = [class_method, instantiate_object]

        variable_0 = TypedVariable(name='file', position=[-198.26100492225922, -124.84927692235689], type_=StringFile)
        variable_1 = TypedVariable(name='end', position=[-227.15499677479085, -73.17410502990185], type_=int)
        variable_2 = TypedVariable(name='remove_duplicates', position=[-227.15499677479085, -45.67410502990185],
                                   type_=bool)
        variable_3 = TypedVariable(name='standalone_subobject', position=[-206.74671747174378, 100.28230774937524],
                                   type_=StandaloneBuiltinsSubobject)
        variable_4 = TypedVariable(name='embedded_subobject', position=[-206.74671747174378, 129.56802203508954],
                                   type_=EmbeddedSubobject)
        variable_5 = TypedVariable(name='dynamic_dict', position=[-206.74671747174378, 158.85373632080382],
                                   type_=Dict[str, str])
        variable_6 = TypedVariable(name='float_dict', position=[-206.74671747174378, 188.1394506065181],
                                   type_=Dict[str, str])
        variable_7 = TypedVariable(name='string_dict', position=[-206.74671747174378, 217.42516489223237],
                                   type_=Dict[str, str])
        variable_8 = TypedVariable(name='object_list', position=[-206.74671747174378, 275.99659346366093],
                                   type_=List[StandaloneBuiltinsSubobject])
        variable_9 = TypedVariable(name='subobject_list', position=[-206.74671747174378, 305.28230774937526],
                                   type_=List[EmbeddedSubobject])
        variable_10 = TypedVariable(name='builtin_list', position=[-206.74671747174378, 334.5680220350895],
                                    type_=List[int])
        variable_11 = TypedVariable(name='subclass_arg', position=[-206.74671747174378, 393.13945060651804],
                                    type_=InstanceOf[StandaloneBuiltinsSubobject])
        variable_12 = TypedVariable(name='array_arg', position=[-206.74671747174378, 422.4251648922324],
                                    type_=List[List[float]])
        variable_13 = TypedVariable(name='name', position=[-206.74671747174378, 451.7108791779467], type_=str)

        pipe_0 = Pipe(variable_0, class_method.inputs[0])
        pipe_1 = Pipe(variable_1, class_method.inputs[1])
        pipe_2 = Pipe(variable_2, class_method.inputs[2])
        pipe_3 = Pipe(variable_3, instantiate_object.inputs[0])
        pipe_4 = Pipe(variable_4, instantiate_object.inputs[1])
        pipe_5 = Pipe(variable_5, instantiate_object.inputs[2])
        pipe_6 = Pipe(variable_6, instantiate_object.inputs[3])
        pipe_7 = Pipe(variable_7, instantiate_object.inputs[4])
        pipe_8 = Pipe(variable_8, instantiate_object.inputs[6])
        pipe_9 = Pipe(variable_9, instantiate_object.inputs[7])
        pipe_10 = Pipe(variable_10, instantiate_object.inputs[8])
        pipe_11 = Pipe(variable_11, instantiate_object.inputs[10])
        pipe_12 = Pipe(variable_12, instantiate_object.inputs[11])
        pipe_13 = Pipe(variable_13, instantiate_object.inputs[12])
        pipes = [pipe_0, pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11,
                 pipe_12, pipe_13]

        workflow = Workflow(blocks, pipes, output=instantiate_object.outputs[0], name='script_workflow')

        expected_script_value = "from dessia_common.typings import ClassMethodType, InstanceOf" \
                                "\nfrom dessia_common.tests import Car" \
                                "\nfrom dessia_common.workflow.blocks import ClassMethod, InstantiateModel" \
                                "\nfrom dessia_common.forms import StandaloneObject, StandaloneBuiltinsSubobject, EmbeddedSubobject" \
                                "\nfrom dessia_common.files import StringFile\nfrom dessia_common.workflow.core import TypedVariable, Pipe, Workflow" \
                                "\nfrom typing import Dict, List" \
                                "\n\n" \
                                "block_0 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='car_from_csv', position=[-27.154996774790852, -138.67410502990185])" \
                                "\nblock_1 = InstantiateModel(model_class=StandaloneObject, name='', position=[-6.746717471743784, 60.49659346366095])" \
                                "\nblocks = [block_0, block_1]\n\nvariable_0 = TypedVariable(name='file', position=[-198.26100492225922, -124.84927692235689], type_=StringFile)" \
                                "\nvariable_1 = TypedVariable(name='end', position=[-227.15499677479085, -73.17410502990185], type_=int)" \
                                "\nvariable_2 = TypedVariable(name='remove_duplicates', position=[-227.15499677479085, -45.67410502990185], type_=bool)" \
                                "\nvariable_3 = TypedVariable(name='standalone_subobject', position=[-206.74671747174378, 100.28230774937524], type_=StandaloneBuiltinsSubobject)" \
                                "\nvariable_4 = TypedVariable(name='embedded_subobject', position=[-206.74671747174378, 129.56802203508954], type_=EmbeddedSubobject)" \
                                "\nvariable_5 = TypedVariable(name='dynamic_dict', position=[-206.74671747174378, 158.85373632080382], type_=Dict[str, str])" \
                                "\nvariable_6 = TypedVariable(name='float_dict', position=[-206.74671747174378, 188.1394506065181], type_=Dict[str, str])" \
                                "\nvariable_7 = TypedVariable(name='string_dict', position=[-206.74671747174378, 217.42516489223237], type_=Dict[str, str])" \
                                "\nvariable_8 = TypedVariable(name='object_list', position=[-206.74671747174378, 275.99659346366093], type_=List[StandaloneBuiltinsSubobject])" \
                                "\nvariable_9 = TypedVariable(name='subobject_list', position=[-206.74671747174378, 305.28230774937526], type_=List[EmbeddedSubobject])" \
                                "\nvariable_10 = TypedVariable(name='builtin_list', position=[-206.74671747174378, 334.5680220350895], type_=List[int])" \
                                "\nvariable_11 = TypedVariable(name='subclass_arg', position=[-206.74671747174378, 393.13945060651804], type_=InstanceOf[StandaloneBuiltinsSubobject])" \
                                "\nvariable_12 = TypedVariable(name='array_arg', position=[-206.74671747174378, 422.4251648922324], type_=List[List[float]])" \
                                "\nvariable_13 = TypedVariable(name='name', position=[-206.74671747174378, 451.7108791779467], type_=str)" \
                                "\n\n" \
                                "pipe_0 = Pipe(variable_0, block_0.inputs[0])" \
                                "\npipe_1 = Pipe(variable_1, block_0.inputs[1])" \
                                "\npipe_2 = Pipe(variable_2, block_0.inputs[2])" \
                                "\npipe_3 = Pipe(variable_3, block_1.inputs[0])" \
                                "\npipe_4 = Pipe(variable_4, block_1.inputs[1])" \
                                "\npipe_5 = Pipe(variable_5, block_1.inputs[2])" \
                                "\npipe_6 = Pipe(variable_6, block_1.inputs[3])" \
                                "\npipe_7 = Pipe(variable_7, block_1.inputs[4])" \
                                "\npipe_8 = Pipe(variable_8, block_1.inputs[6])" \
                                "\npipe_9 = Pipe(variable_9, block_1.inputs[7])" \
                                "\npipe_10 = Pipe(variable_10, block_1.inputs[8])" \
                                "\npipe_11 = Pipe(variable_11, block_1.inputs[10])" \
                                "\npipe_12 = Pipe(variable_12, block_1.inputs[11])" \
                                "\npipe_13 = Pipe(variable_13, block_1.inputs[12])" \
                                "\npipes = [pipe_0, pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8, pipe_9, pipe_10, pipe_11, pipe_12, pipe_13]" \
                                "\n\n" \
                                "workflow = Workflow(blocks, pipes, output=block_1.outputs[0], name='script_workflow')\n"
        self.assertEqual(workflow.to_script(), expected_script_value)