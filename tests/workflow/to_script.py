
import dessia_common.typings as dct
import dessia_common.workflow as wf
import dessia_common.tests as dctests

import tests.workflow.workflow_with_models as model_test
from dessia_common import DessiaFilter

model_test.demo_workflow.to_script()

test_blocks = [
    wf.ForEach(model_test.optimization_workflow_block, 0),
    wf.Archive(),
    wf.ClassMethod(
        method_type=dct.ClassMethodType(dctests.Car, 'from_csv'),
        name='car_from_csv'
    ),
    wf.InstantiateModel(dctests.Car, name='Instantiate Car'),
    wf.ModelAttribute('model_to_optimize', name='Model Fetcher'),
    wf.ModelMethod(
        method_type=dct.MethodType(dctests.Car, 'to_vector'),
        name='car_to_vector'
    ),
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
    wf.Export(
        method_type=dct.MethodType(dctests.Car, 'to_vector'),
        name='Export',
        export_name="export_name"),
]

test_pipes = [wf.Pipe(test_blocks[0].outputs[0], test_blocks[3].inputs[0])]

workflow_script = wf.Workflow(test_blocks, test_pipes, test_blocks[0].outputs[0], name="script_workflow")
workflow_script.to_script()
