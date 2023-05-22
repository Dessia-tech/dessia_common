# import unittest
#
# import dessia_common.typings as dct
# from dessia_common.workflow.core import Pipe, Workflow
# from dessia_common.workflow.blocks import InstantiateModel, ModelMethod, ModelAttribute, WorkflowBlock, ForEach,\
#     Archive, ClassMethod, Sequence, SetModelAttribute, Substraction, Sum, Flatten, Filter, Unpacker, Product,\
#     MultiPlot, Export, GetModelAttribute
# import dessia_common.tests as dctests
#
# from dessia_common.core import DessiaFilter


# class WorkflowToScriptTest(unittest.TestCase):

    # def test_simple_equality(self):
    #     instantiate_optimizer = InstantiateModel(model_class=dctests.Optimizer, name='Instantiate Optimizer')
    #     optimization = ModelMethod(dct.MethodType(dctests.Optimizer, 'optimize'), name='Optimization')
    #     model_fetcher = ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')
    #     optimization_blocks = [instantiate_optimizer, optimization, model_fetcher]
    #
    #     pipe1_opt = Pipe(input_variable=instantiate_optimizer.outputs[0], output_variable=optimization.inputs[0])
    #     pipe2_opt = Pipe(input_variable=optimization.outputs[1], output_variable=model_fetcher.inputs[0])
    #     optimization_pipes = [pipe1_opt, pipe2_opt]
    #     optimization_workflow = Workflow(blocks=optimization_blocks, pipes=optimization_pipes,
    #                                      output=model_fetcher.outputs[0], name="Optimization's Workflow")
    #     optimization_workflow_block = WorkflowBlock(workflow=optimization_workflow, name='Workflow Block')
    #
    #     blocks = [
    #         ForEach(optimization_workflow_block, 0, position=(11.11, 22)),
    #         Archive(position=(22.22, 33)),
    #         ClassMethod(
    #             method_type=dct.ClassMethodType(
    #                 dctests.Car, 'from_csv'), name='car_from_csv', position=(
    #                 33.33, 44)),
    #         InstantiateModel(dctests.Car, name='Instantiate Car', position=(44.44, 55)),
    #         ModelAttribute('model_to_optimize', name='Model Fetcher', position=(55.55, 66)),
    #         ModelMethod(
    #             method_type=dct.MethodType(
    #                 dctests.Car,
    #                 'to_vector'),
    #             name='car_to_vector',
    #             position=(
    #                 66.66,
    #                 77)),
    #         Sequence(3, "sequence_name", position=(77.77, 88)),
    #         SetModelAttribute(dct.AttributeType(dctests.Optimizer, name='model_to_optimize'), 'name_Name', position=(88.88, 99)),
    #         GetModelAttribute(dct.AttributeType(dctests.Optimizer, name='model_to_optimize'), 'name_Name', position=(12.34, 56.78)),
    #         Substraction("substraction_name", position=(99.99, 11.11)),
    #         Sum(name="sum_name", position=(22, 11.11)),
    #         Flatten('flatten_name', position=(33, 22.22)),
    #         Filter([DessiaFilter("attributeFilter", "operatorFilter", bound=3.1415)], position=(44, 33.33)),
    #         Unpacker([1, 3], "unpacker_name", position=(55, 44.44)),
    #         MultiPlot(['multiplot0', 'multiplot1'], position=(77, 66.66)),
    #         Product(4, "product_name", position=(88, 77.77)),
    #         Export(method_type=dct.MethodType(dctests.Model, 'save_to_stream'), name='Export', filename="filename",
    #                extension="json", text=True, position=(99, 88.88))
    #     ]
    #
    #     pipes = [Pipe(blocks[0].outputs[0], blocks[3].inputs[0])]
    #
    #     workflow = Workflow(blocks, pipes, blocks[0].outputs[0], name="script_workflow")

        # expected_script_value = "from dessia_common.tests import Optimizer, Car, Model" \
        #    "\nfrom dessia_common.workflow.blocks import InstantiateModel, ModelMethod, ModelAttribute, WorkflowBlock, ForEach, Archive, ClassMethod, Sequence, SetModelAttribute, GetModelAttribute, Substraction, Sum, Flatten, Filter, Unpacker, MultiPlot, Product, Export" \
        #    "\nfrom dessia_common.typings import MethodType, ClassMethodType, AttributeType" \
        #    "\nfrom dessia_common.workflow.core import Pipe, Workflow" \
        #    "\nfrom dessia_common.core import DessiaFilter" \
        #    "\n" \
        #    "\n" \
        #    "\n# --- Subworkflow --- " \
        #    "\nsub_block_0 = InstantiateModel(model_class=Optimizer, name=\"Instantiate Optimizer\", position=(0, 0))" \
        #    "\nsub_block_1 = ModelMethod(method_type=MethodType(Optimizer, 'optimize'), name=\"Optimization\", position=(0, 0))" \
        #    "\nsub_block_2 = ModelAttribute(attribute_name='model_to_optimize', name=\"Model Fetcher\", position=(0, 0))" \
        #    "\nsub_blocks = [sub_block_0, sub_block_1, sub_block_2]" \
        #    "\n\n" \
        #    "\nsub_pipe_0 = Pipe(sub_block_0.outputs[0], sub_block_1.inputs[0])" \
        #    "\nsub_pipe_1 = Pipe(sub_block_1.outputs[1], sub_block_2.inputs[0])" \
        #    "\nsub_pipes = [sub_pipe_0, sub_pipe_1]" \
        #    "\n" \
        #    "\nsub_workflow = Workflow(sub_blocks, sub_pipes, output=sub_block_2.outputs[0], name=\"Optimization's Workflow\")" \
        #    "\n# --- End Subworkflow --- " \
        #    "\n" \
        #    "\nblock_7 = SetModelAttribute(attribute_type=AttributeType(Optimizer, name=\"model_to_optimize\"), name=\"name_Name\", position=(88.88, 99))" \
        #    "\nblock_8 = GetModelAttribute(attribute_type=AttributeType(Optimizer, name=\"model_to_optimize\"), name=\"name_Name\", position=(12.34, 56.78))" \
        #    "\nblock_9 = Substraction(name=\"substraction_name\", position=(99.99, 11.11))" \
        #    "\nblock_10 = Sum(number_elements=2, name=\"sum_name\", position=(22, 11.11))" \
        #    "\nblock_11 = Flatten(name=\"flatten_name\", position=(33, 22.22))" \
        #    "\nblock_12 = Filter(filters=[DessiaFilter(attribute='attributeFilter', comparison_operator='operatorFilter', bound=3.1415, name='')], logical_operator='and', name=\"\", position=(44, 33.33))" \
        #    "\nblock_13 = Unpacker(indices=[1, 3], name=\"unpacker_name\", position=(55, 44.44))" \
        #    "\nblock_14 = MultiPlot(attributes=['multiplot0', 'multiplot1'], name=\"\", position=(77, 66.66))" \
        #    "\nblock_15 = Product(number_list=4, name=\"product_name\", position=(88, 77.77))" \
        #    "\nblock_16 = Export(method_type=MethodType(Model, 'save_to_stream'), filename='filename', extension='json', text=True, name=\"Export\", position=(99, 88.88))" \
        #    "\nblocks = [block_0, block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9, block_10, block_11, block_12, block_13, block_14, block_15, block_16]" \
        #    "\n\n" \
        #    "\npipe_0 = Pipe(block_0.outputs[0], block_3.inputs[0])" \
        #    "\npipes = [pipe_0]" \
        #    "\n" \
        #    "\nworkflow = Workflow(blocks, pipes, output=block_0.outputs[0], name=\"script_workflow\")" \
        #    "\n"
        # self.assertEqual(workflow.to_script(), expected_script_value)
