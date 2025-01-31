import unittest
from parameterized import parameterized
from dessia_common.workflow.core import Workflow, Pipe
from dessia_common.workflow.blocks import InstantiateModel, ModelMethod
from dessia_common.forms import Generator, StandaloneObject
from dessia_common.typings import MethodType
from dessia_common.displays import DisplaySetting

VIEW_SETTING = DisplaySetting(selector="2D View", type_="plot_data", method="primitives", serialize_data=True)
MD_SETTING = DisplaySetting(selector="Markdown", type_="markdown", method="to_markdown", load_by_default=True)


class TestDisplays(unittest.TestCase):

    def setUp(self):
        generator = InstantiateModel(model_class=Generator, name="Generator")
        generate_method = MethodType(class_=Generator, name='generate')
        generate = ModelMethod(method_type=generate_method, name='Generate')
        count_method = MethodType(class_=StandaloneObject, name='count_until')
        count = ModelMethod(method_type=count_method, name="Count")
        blocks = [generator, generate, count]
        pipes = [Pipe(generator.outputs[0], generate.inputs[0]),
                 Pipe(generate.outputs[0], count.inputs[0])]
        self.workflow = Workflow(blocks=blocks, pipes=pipes, output=count.outputs[0])
        self.workflow.insert_step("A")

    def test_displayable_outputs(self):
        self.assertEqual(len(self.workflow.displayable_outputs), 3)

    @parameterized.expand([
            (0, 4),
            (1, 4),
            (2, 6),
        ])
    def test_displayable_upstreams(self, output_index: int, expected_upstreams_length: int):
        output = self.workflow.displayable_outputs[output_index]
        upstreams = self.workflow.upstream_inputs(output)
        self.assertEqual(len(upstreams), expected_upstreams_length)

    @parameterized.expand([
        (12, "2D View", VIEW_SETTING, 6),
        (7, "Markdown", MD_SETTING, 4)
    ])
    def test_step_display(self, variable_index: int, selector: str,
                          expected_display_setting: DisplaySetting, expected_group_inputs_length: int):
        step_index = 0
        self.workflow.add_step_display(variable_index=variable_index, step_index=step_index, selector=selector)
        step = self.workflow.steps[step_index]
        self.assertEqual(step.display_setting, expected_display_setting)
        self.assertEqual(len(step.group_inputs), expected_group_inputs_length)

    def test_remove_display(self):
        step_index = 0
        self.workflow.add_step_display(variable_index=12, step_index=step_index, selector="2D View")
        self.workflow.remove_step_display(step_index)
        step = self.workflow.steps[step_index]
        self.assertIsNone(step.display_setting)
        self.assertEqual(len(step.group_inputs), 0)
