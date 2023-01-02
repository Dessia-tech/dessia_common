import unittest

    
class BackendBreakingChangeTest(unittest.TestCase):
    def test_import_is_working(self):
        """Basic unittest to make sure backend import of DC is working"""
        try:
            from dessia_common import __version__
            from dessia_common.core import DessiaObject, inspect_arguments, stringify_dict_keys
            from dessia_common.errors import DeepAttributeError
            from dessia_common.files import BinaryFile, StringFile
            from dessia_common.utils.jsonschema import default_dict
            from dessia_common.utils.serialization import serialize, serialize_with_pointers
            from dessia_common.utils.types import TYPING_EQUIVALENCES, is_bson_valid, is_jsonable, is_sequence, serialize_typing
            from dessia_common.workflow.core import WorkflowState        
        except:
            self.fail("Breaking changes for backend detected. Please create a backend ticket with changes for fixing this UT.")
