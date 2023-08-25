import unittest
from parameterized import parameterized
from dessia_common import REF_MARKER
from dessia_common.forms import EmbeddedSubobject, StandaloneObject, EnhancedEmbeddedSubobject,\
    DEF_SO, DEF_SOWDV
from dessia_common.utils.helpers import full_classname


class TestFeatures(unittest.TestCase):

    def test_viability(self):
        DEF_SO._check_platform()

    @parameterized.expand([
        ("#/standalone_subobject/floatarg", 0.3),
        ("#/embedded_subobject/embedded_list/2", 3),
        ("#/object_list/0/floatarg", 0.3),
        ("#/standalone_subobject/name", "EmbeddedSubobject1"),
        ("name", "Standalone Object Demo")
    ])
    def test_deep_attributes(self, attribute, expected_value):
        self.assertEqual(DEF_SO._get_from_path(attribute), expected_value)

    @parameterized.expand([
        (EmbeddedSubobject, "dessia_common.forms.EmbeddedSubobject"),
        (EnhancedEmbeddedSubobject, "dessia_common.forms.EnhancedEmbeddedSubobject")
    ])
    def test_classnames(self, class_, expected_name):
        classname = full_classname(object_=class_, compute_for='class')
        self.assertEqual(classname, expected_name)

    @parameterized.expand([
        (DEF_SO, "dessia_common.forms.StandaloneObject"),
        (DEF_SOWDV, "dessia_common.forms.StandaloneObjectWithDefaultValues")
    ])
    def test_object_classnames(self, object_, expected_name):
        classname = full_classname(object_=object_, compute_for='object_')
        self.assertEqual(classname, expected_name)

    def test_serialization(self):
        dict_ = DEF_SO.to_dict(use_pointers=True)
        self.assertIn(REF_MARKER, dict_['union_arg'][1])
        obj = StandaloneObject.dict_to_object(dict_)
        assert not isinstance(obj.union_arg[1], dict)

    def test_xlsx_export(self):
        DEF_SO.to_xlsx('test')
