import unittest
from tests.test_serialization.non_dessia_object_subobjects import container, line
from dessia_common import REF_MARKER


class TestNonDessiaObjects(unittest.TestCase):

    def test_container(self):
        # container._check_platform()
        dict_ = container.to_dict()
        self.assertIn(REF_MARKER, dict_["points"][0])
        self.assertIn("#/_references/", dict_["points"][0][REF_MARKER])
        self.assertEqual(dict_["points"][1], container.points[1].to_dict())
        self.assertEqual(dict_["points"][2], {REF_MARKER: "#/points/0"})

    def test_line(self):
        # line._check_platform()
        dict_ = line.to_dict()
        self.assertIn(REF_MARKER, dict_["p1"])
        self.assertIn("#/_references/", dict_["p1"][REF_MARKER])
        self.assertEqual(dict_["p2"], {REF_MARKER: "#/p1"})
