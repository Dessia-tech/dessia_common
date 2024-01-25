import unittest
from typing import Dict, List, Tuple, Union

from dessia_common.files import BinaryFile
from dessia_common.forms import EmbeddedSubobject, StandaloneBuiltinsSubobject
from dessia_common.measures import Measure
from dessia_common.schemas.core import (BuiltinProperty, DynamicDict,
                                        HeterogeneousSequence,
                                        HomogeneousSequence,
                                        InstanceOfProperty, MeasureProperty,
                                        SchemaAttribute, UnionProperty)
from dessia_common.typings import InstanceOf
from parameterized import parameterized

schemas = [
    (BuiltinProperty(annotation=str, attribute=SchemaAttribute(name="BuiltinProperty")), []),
    (MeasureProperty(annotation=Measure, attribute=SchemaAttribute(name="MeasureProperty")),
     ["dessia_common.measures.Measure"]),

    (UnionProperty(annotation=Union[EmbeddedSubobject, StandaloneBuiltinsSubobject],
                   attribute=SchemaAttribute(name="UnionProperty")),
     ['typing.Union', 'dessia_common.forms.EmbeddedSubobject', 'dessia_common.forms.StandaloneBuiltinsSubobject']),

    (HeterogeneousSequence(annotation=Tuple[BinaryFile, BinaryFile], attribute=SchemaAttribute("HeterogeneousSequence")),
     ['typing.Tuple', 'dessia_common.files.BinaryFile', 'dessia_common.files.BinaryFile']
     ),

    (HomogeneousSequence(annotation=List[Tuple[bool]], attribute=SchemaAttribute(name="HomogeneousSequence")),
     ['typing.List', 'typing.Tuple']),

    (DynamicDict(annotation=Dict[str, int], attribute=SchemaAttribute(name="DynamicDict")),
     ['typing.Dict']),

    (HomogeneousSequence(annotation=List[List[List[Tuple[float, float]]]],
                         attribute=SchemaAttribute(name="HomogeneousSequence")),
     ['typing.List', 'typing.List', 'typing.List', 'typing.Tuple']),

    (InstanceOfProperty(annotation=InstanceOf[EmbeddedSubobject],
                         attribute=SchemaAttribute(name="InstanceOfProperty")),
     ['dessia_common.typings.InstanceOf', 'dessia_common.forms.EmbeddedSubobject']),

]


class TestGetNames(unittest.TestCase):

    @parameterized.expand(schemas)
    def test_get_names(self, schema, expected_declaration):
        self.assertEqual(schema.get_import_names([]), expected_declaration)


if __name__ == '__main__':
    unittest.main(verbosity=2)
