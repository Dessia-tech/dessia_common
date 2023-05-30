from dessia_common.schemas.core import UnionProperty, InstanceOfProperty, SubclassProperty
from dessia_common.forms import EmbeddedSubobject, StandaloneBuiltinsSubobject, StandaloneObject, Generator
from dessia_common.typings import InstanceOf
from typing import Union

import unittest
from parameterized import parameterized


class TestInheritance(unittest.TestCase):
    @parameterized.expand([
        (
            InstanceOfProperty(annotation=InstanceOf[EmbeddedSubobject], attribute="instance_of"),
            "InstanceOf[dessia_common.forms.EmbeddedSubobject]",
            False,
            ["dessia_common.forms.EmbeddedSubobject"]
        ),
        (
            InstanceOfProperty(annotation=InstanceOf[StandaloneBuiltinsSubobject], attribute="instance_of"),
            "InstanceOf[dessia_common.forms.StandaloneBuiltinsSubobject]",
            True,
            ["dessia_common.forms.StandaloneBuiltinsSubobject"]
        ),
        (
            UnionProperty(annotation=Union[EmbeddedSubobject, StandaloneBuiltinsSubobject], attribute="union"),
            "Union[dessia_common.forms.EmbeddedSubobject, dessia_common.forms.StandaloneBuiltinsSubobject]",
            None,
            ["dessia_common.forms.EmbeddedSubobject", "dessia_common.forms.StandaloneBuiltinsSubobject"]
        ),
        (
            UnionProperty(annotation=Union[StandaloneObject, Generator], attribute="union"),
            "Union[dessia_common.forms.StandaloneObject, dessia_common.forms.Generator]",
            True,
            ["dessia_common.forms.StandaloneObject", "dessia_common.forms.Generator"]
        )
    ])
    def test_unions(self, schema, expected_typing, expected_standalone, expected_classes):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], "object")
        self.assertEqual(computed_schema["python_typing"], expected_typing)
        self.assertEqual(computed_schema["classes"], expected_classes)
        self.assertEqual(computed_schema["standalone_in_db"], expected_standalone)


if __name__ == '__main__':
    unittest.main(verbosity=2)
