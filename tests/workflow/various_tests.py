import unittest

from dessia_common.forms import StandaloneObjectWithDefaultValues, StandaloneObject
from dessia_common.workflow import InstantiateModel, Workflow, WorkflowError


class WorkflowTests(unittest.TestCase):

    def test_output_in_init(self):
        print("Making sure that output must be valid in Workflow.__init__()")
        instantiate = InstantiateModel(model_class=StandaloneObjectWithDefaultValues, name='Instantiate SOWDV')
        instantiate_no_dv = InstantiateModel(model_class=StandaloneObject, name='Instantiate No DV')

        # Should not raise an error
        Workflow(
            blocks=[instantiate, instantiate_no_dv],
            pipes=[],
            output=instantiate.outputs[0]
        )

        # Following asserts are OK iff Workflow.__init__ raises an error
        # as output is not valid
        self.assertRaises(
            WorkflowError,
            Workflow,
            name="SOWDV",
            blocks=[instantiate, instantiate_no_dv],
            pipes=[],
            output=instantiate.inputs[0]
        )
        self.assertRaises(
            WorkflowError,
            Workflow,
            blocks=[instantiate_no_dv],
            pipes=[],
            output=instantiate.outputs[0]
        )