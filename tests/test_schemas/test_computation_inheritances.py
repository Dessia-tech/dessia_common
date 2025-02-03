from dessia_common.schemas.core import UnionProperty, InstanceOfProperty, SchemaAttribute
from dessia_common.forms import EmbeddedSubobject, StandaloneBuiltinsSubobject, StandaloneObject, Generator
from dessia_common.typings import InstanceOf
from typing import Union

import unittest
from parameterized import parameterized

EMBEDDED_ATTRIBUTE = SchemaAttribute(name="instance_of")
STANDALONE_ATTRIBUTE = SchemaAttribute(name="instance_of")
EMB_UNION_ATTR = SchemaAttribute(name="union")
STANDALONE_UNION_ATTRIBUTE = SchemaAttribute(name="union")


class TestInheritance(unittest.TestCase):
    @parameterized.expand([
        (
            InstanceOfProperty(annotation=InstanceOf[EmbeddedSubobject], attribute=EMBEDDED_ATTRIBUTE),
            "InstanceOf[dessia_common.forms.EmbeddedSubobject]",
            False,
            ["dessia_common.forms.EmbeddedSubobject"]
        ),
        (
            InstanceOfProperty(annotation=InstanceOf[StandaloneBuiltinsSubobject], attribute=STANDALONE_ATTRIBUTE),
            "InstanceOf[dessia_common.forms.StandaloneBuiltinsSubobject]",
            True,
            ["dessia_common.forms.StandaloneBuiltinsSubobject"]
        ),
        (
            UnionProperty(annotation=Union[EmbeddedSubobject, StandaloneBuiltinsSubobject], attribute=EMB_UNION_ATTR),
            "Union[dessia_common.forms.EmbeddedSubobject, dessia_common.forms.StandaloneBuiltinsSubobject]",
            None,
            ["dessia_common.forms.EmbeddedSubobject", "dessia_common.forms.StandaloneBuiltinsSubobject"]
        ),
        (
            UnionProperty(annotation=Union[StandaloneObject, Generator], attribute=STANDALONE_UNION_ATTRIBUTE),
            "Union[dessia_common.forms.StandaloneObject, dessia_common.forms.Generator]",
            True,
            ["dessia_common.forms.StandaloneObject", "dessia_common.forms.Generator"]
        )
    ])
    def test_unions(self, schema, expected_typing, expected_standalone, expected_classes):
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["type"], "object")
        self.assertEqual(computed_schema["pythonTyping"], expected_typing)
        self.assertEqual(computed_schema["classes"], expected_classes)
        self.assertEqual(computed_schema["standaloneInDb"], expected_standalone)


if __name__ == '__main__':
    unittest.main(verbosity=2)
