
import dessia_common.typings as dct
import dessia_common.workflow as wf
import dessia_common.tests as dctests

from dessia_common import DessiaFilter


instanciate_optimizer = wf.InstantiateModel(model_class=dctests.Optimizer, name='Instantiate Optimizer')
optimization = wf.ModelMethod(dct.MethodType(dctests.Optimizer, 'optimize'), name='Optimization')
model_fetcher = wf.ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')
optimization_blocks = [instanciate_optimizer, optimization, model_fetcher]

pipe1_opt = wf.Pipe(input_variable=instanciate_optimizer.outputs[0], output_variable=optimization.inputs[0])
pipe2_opt = wf.Pipe(input_variable=optimization.outputs[1], output_variable=model_fetcher.inputs[0])
optimization_pipes = [pipe1_opt, pipe2_opt]
optimization_workflow = wf.Workflow(blocks=optimization_blocks, pipes=optimization_pipes,
                                    output=model_fetcher.outputs[0], name='Optimization Workflow')
optimization_workflow_block = wf.WorkflowBlock(workflow=optimization_workflow, name='Workflow Block')


test_blocks = [
    wf.ForEach(optimization_workflow_block, 0),
    wf.Archive(),
    wf.ClassMethod(method_type=dct.ClassMethodType(dctests.Car, 'from_csv'), name='car_from_csv'),
    wf.InstantiateModel(dctests.Car, name='Instantiate Car'),
    wf.ModelAttribute('model_to_optimize', name='Model Fetcher'),
    wf.ModelMethod(method_type=dct.MethodType(dctests.Car, 'to_vector'), name='car_to_vector'),
    wf.Sequence(3, "sequence_name"),
    wf.SetModelAttribute('name', 'name_Name'),
    wf.Substraction("substraction_name"),
    wf.Sum(name="sum_name"),
    wf.Flatten('flatten_name'),
    wf.Filter([DessiaFilter("attributeFilter", "operatorFilter", bound=3.1415)]),
    wf.Unpacker([1, 3], "unpacker_name"),
    wf.Display(),
    wf.MultiPlot(['multiplot0', 'multiplot1']),
    wf.Product(4, "product_name"),
    wf.Export(method_type=dct.MethodType(dctests.Model, 'save_to_stream'), name='Export', export_name="export_name",
              extension="json", text=True)
]

test_pipes = [wf.Pipe(test_blocks[0].outputs[0], test_blocks[3].inputs[0])]

workflow_script = wf.Workflow(test_blocks, test_pipes, test_blocks[0].outputs[0], name="script_workflow")

script_value = "from dessia_common.tests import Optimizer, Car, Model" \
               "\nfrom dessia_common.workflow.blocks import InstantiateModel, ModelMethod, ModelAttribute, WorkflowBlock, ForEach, Archive, ClassMethod, Sequence, SetModelAttribute, Substraction, Sum, Flatten, Filter, Unpacker, Display, MultiPlot, Product, Export" \
               "\nfrom dessia_common.typings import MethodType, ClassMethodType" \
               "\nfrom dessia_common.core import DessiaFilter" \
               "\nfrom dessia_common.workflow.core import Variable, Workflow, Pipe" \
               "\n" \
               "\n" \
               "\n# --- Subworkflow --- " \
               "\nsub_block_0 = InstantiateModel(model_class=Optimizer, name='Instantiate Optimizer')" \
               "\nsub_block_1 = ModelMethod(method_type=MethodType(Optimizer, 'optimize'), name='Optimization')" \
               "\nsub_block_2 = ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')" \
               "\nsub_blocks = [sub_block_0, sub_block_1, sub_block_2]" \
               "\n" \
               "\nsub_pipe_0 = Pipe(sub_block_0.outputs[0], sub_block_1.inputs[0])" \
               "\nsub_pipe_1 = Pipe(sub_block_1.outputs[1], sub_block_2.inputs[0])" \
               "\nsub_pipes = [sub_pipe_0, sub_pipe_1]" \
               "\n" \
               "\nsub_workflow = Workflow(sub_blocks, sub_pipes, output=sub_block_2.outputs[0], name='Optimization Workflow')" \
               "\n# --- End Subworkflow --- " \
               "\n" \
               "\nwfblock = WorkflowBlock(workflow=sub_workflow, name='Workflow Block')" \
               "\nblock_0 = ForEach(workflow_block=wfblock, iter_input_index=0)" \
               "\nblock_1 = Archive(number_exports=1, export_name='archive', name='')" \
               "\nblock_2 = ClassMethod(method_type=ClassMethodType(Car, 'from_csv'), name='car_from_csv')" \
               "\nblock_3 = InstantiateModel(model_class=Car, name='Instantiate Car')" \
               "\nblock_4 = ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')" \
               "\nblock_5 = ModelMethod(method_type=MethodType(Car, 'to_vector'), name='car_to_vector')" \
               "\nblock_6 = Sequence(number_arguments=3, name='sequence_name')" \
               "\nblock_7 = SetModelAttribute(attribute_name='name', name='name_Name')" \
               "\nblock_8 = Substraction(name='substraction_name')" \
               "\nblock_9 = Sum(number_elements=2, name='sum_name')" \
               "\nblock_10 = Flatten(name='flatten_name')" \
               "\nblock_11 = Filter(filters=[DessiaFilter(attribute='attributeFilter', operator='operatorFilter', bound=3.1415, name='')], name='')" \
               "\nblock_12 = Unpacker(indices=[1, 3], name='unpacker_name')" \
               "\nblock_13 = Display(inputs=None, name='')" \
               "\nblock_14 = MultiPlot(attributes=['multiplot0', 'multiplot1'], name='')" \
               "\nblock_15 = Product(number_list=4, name='product_name')" \
               "\nblock_16 = Export(method_type=MethodType(dessia_common.tests.Model, 'save_to_stream'), export_name='export_name', extension='json', text=True, name='Export')" \
               "\nblocks = [block_0, block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9, block_10, block_11, block_12, block_13, block_14, block_15, block_16]" \
               "\n" \
               "\npipe_0 = Pipe(block_0.outputs[0], block_3.inputs[0])" \
               "\npipes = [pipe_0]" \
               "\n" \
               "\nworkflow = Workflow(blocks, pipes, output=block_0.outputs[0], name='script_workflow')" \
               "\n"


assert workflow_script.to_script() == script_value
print("test to_script.py passed")
