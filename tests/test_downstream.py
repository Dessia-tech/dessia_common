import unittest


class BackendBreakingChangeTest(unittest.TestCase):
    def test_import_is_working(self):
        """Basic unittest to make sure backend import of DC is working"""
        try:
            from dessia_common import __version__
            from dessia_common.core import DessiaObject, stringify_dict_keys
            from dessia_common.errors import DeepAttributeError
            from dessia_common.files import BinaryFile, StringFile
            from dessia_common.serialization import serialize, serialize_with_pointers
            from dessia_common.utils.helpers import is_sequence
            from dessia_common.utils.types import is_bson_valid, is_jsonable
            from dessia_common.workflow.core import WorkflowRun, WorkflowState
            from dessia_common.workflow.utils import ToScriptElement
            from dessia_common.schemas.core import (
                TYPING_EQUIVALENCES,
                ClassSchema,
                MethodSchema,
                Schema,
                inspect_arguments,
                serialize_annotation,
            )
        except Exception:
            self.fail("Breaking changes for backend detected."
                      "Please create a backend ticket with changes for fixing this UT.")
