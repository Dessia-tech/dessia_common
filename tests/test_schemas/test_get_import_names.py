from typing import List, Tuple, Union

from dessia_common.measures import Measure
from dessia_common.schemas.core import ClassProperty, MethodTypeProperty, AttributeTypeProperty, SchemaAttribute,\
    UnionProperty, HeterogeneousSequence, HomogeneousSequence, BuiltinProperty, MeasureProperty
from dessia_common.forms import StandaloneObject, EmbeddedSubobject, StandaloneBuiltinsSubobject
from dessia_common.typings import MethodType, ClassMethodType, AttributeType, ClassAttributeType
from typing import Type


import unittest
from parameterized import parameterized


schemas = [
    (BuiltinProperty(annotation=str, attribute=SchemaAttribute(name="BuiltinProperty")), []),
    (MeasureProperty(annotation=Measure, attribute=SchemaAttribute(name="MeasureProperty")),
     ["dessia_common.measures.Measure"]),

    (UnionProperty(annotation=Union[EmbeddedSubobject, StandaloneBuiltinsSubobject],
                   attribute=SchemaAttribute(name="UnionProperty")),
     ['typing.Union', 'dessia_common.forms.EmbeddedSubobject', 'dessia_common.forms.StandaloneBuiltinsSubobject']),

    (HomogeneousSequence(annotation=List[Tuple[bool]], attribute=SchemaAttribute(name="HomogeneousSequence")),
     ['typing.List', 'typing.Tuple']),
    # (), (), (), ()
]


class TestGetNames(unittest.TestCase):

    @parameterized.expand(schemas)
    def test_get_names(self, schema, expected_declaration):
        self.assertEqual(schema.get_import_names([]), expected_declaration)


if __name__ == '__main__':
    unittest.main(verbosity=2)
