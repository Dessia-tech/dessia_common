from dessia_common.forms import Optimizer, StandaloneObject
from dessia_common.typings import MethodType
from dessia_common.workflow.core import Pipe, Workflow
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod, ModelAttribute

import unittest
from parameterized import parameterized

# ----- Workflow declaration -----
instantiate = InstantiateModel(model_class=Optimizer, name='Instantiate Optimizer')
optimize = ModelMethod(
    method_type=MethodType(class_=Optimizer, name='optimize'),
    name="Optimization")
model_fetcher = ModelAttribute(attribute_name='model_to_optimize', name='Model Fetcher')

blocks = [instantiate, optimize, model_fetcher]
pipes = [
    Pipe(input_variable=instantiate.outputs[0], output_variable=optimize.inputs[0]),
    Pipe(input_variable=optimize.outputs[0], output_variable=model_fetcher.inputs[0])
]

optimization_workflow = Workflow(blocks=blocks, pipes=pipes, output=model_fetcher.outputs[0], name="DC- Opti workflow")

optimization_workflow2 = optimization_workflow.copy()
optimization_workflow2.imposed_variable_values[optimization_workflow2.blocks[0].inputs[1]] = "custom_name"

generated_standalone_object = StandaloneObject.generate(1)
optimization_workflow3 = optimization_workflow2.copy()
optimization_workflow3.imposed_variable_values[optimization_workflow3.blocks[0].inputs[0]] = generated_standalone_object


class TestMethodDict(unittest.TestCase):

    @parameterized.expand([
        (optimization_workflow, {1: "", 2: 3}),
        (optimization_workflow2, {1: "custom_name", 2: 3}),
        (optimization_workflow3, {0: generated_standalone_object, 1: "custom_name", 2: 3})
    ])
    def test_method_dict_is_valid(self, workflow, expected_dict):
        self.assertEqual(expected_dict, workflow.method_dict(method_name='run'))
