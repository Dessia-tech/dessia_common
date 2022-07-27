from dessia_common.forms import Optimizer, StandaloneObject
from dessia_common.typings import MethodType
from dessia_common.workflow import InstantiateModel, ModelMethod, ModelAttribute, Pipe, Workflow


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


# ----- Utils -----
def assert_str(case: str) -> str:
    return f"\n" \
           f"-- Failure : {case}\n" \
           f"-- Got : {optimization_workflow.method_dict(method_name='run')}\n" \
           f"-- Expected : {expected_dict}"


# ----- Tests -----
expected_dict = {
    1: "",
    2: 3
}
assert optimization_workflow.method_dict(method_name='run') == expected_dict, \
    assert_str('Basic method_dict')

optimization_workflow.imposed_variable_values[instantiate.inputs[1]] = "custom_name"
expected_dict[1] = "custom_name"
assert optimization_workflow.method_dict(method_name='run') == expected_dict, \
    assert_str('imposed_variable_value on optional variable')

SO = StandaloneObject.generate(1)
optimization_workflow.imposed_variable_values[instantiate.inputs[0]] = SO
expected_dict[0] = SO
assert optimization_workflow.method_dict(method_name='run') == expected_dict, \
    assert_str("imposed_variable_value on required variable")
