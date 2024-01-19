import unittest
from dessia_common import REF_MARKER
from dessia_common.core import DessiaObject
from typing import List


class Point:

    def __init__(self, name: str = ""):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class Vector:

    def __init__(self, name: str = ""):
        self.name = name

    def to_dict(self):
        return {"name": self.name}

    def __eq__(self, other):
        return True

    def __hash__(self):
        return 0


class Module(DessiaObject):
    _standalone_in_db = True

    def __init__(self, direction: Vector, origin: Point, name: str = ''):
        self.origin = origin
        self.direction = direction

        DessiaObject.__init__(self, name=name)


class Vessel(DessiaObject):
    _standalone_in_db = True

    def __init__(self, modules: List[Module], name: str = ''):
        self.modules = modules

        DessiaObject.__init__(self, name=name)


class Layout(DessiaObject):
    _standalone_in_db = True

    def __init__(self, vessel: Vessel, name: str = ''):
        self.vessel = vessel
        DessiaObject.__init__(self, name=name)


VECTORS = [Vector("V1"), Vector("V2")]
POINTS = [Point("P1"), Point("P2")]

MODULES = [Module(origin=p, direction=v, name=f"Module {i + 1}") for i, (p, v) in enumerate(zip(POINTS, VECTORS))]
VESSEL = Vessel(modules=MODULES, name="Vessel")
LAYOUT = Layout(vessel=VESSEL, name="Layout")


class TestNonDessiaObjectsStructure(unittest.TestCase):

    def setUp(self):
        self.dict_ = LAYOUT.to_dict()

    def test_serialize(self):
        ref_prefix = "#/_references/"
        expected_vector1_dict = {"name": "V1"}
        references = self.dict_["_references"]
        # Test Vessel
        self.assertIn(REF_MARKER, self.dict_["vessel"])
        self.assertIn(ref_prefix, self.dict_["vessel"][REF_MARKER])
        vessel_id = self.dict_["vessel"][REF_MARKER].split(ref_prefix)[1]
        self.assertIn(vessel_id, references)
        vessel_dict = references[vessel_id]

        # Test Module 1
        module1_id = vessel_dict["modules"][0][REF_MARKER].split(ref_prefix)[1]
        module1_dict = references[module1_id]

        # Test Vector 1
        self.assertEqual(module1_dict["direction"], expected_vector1_dict)
        # TODO Assertion above should actually be a reference (like module2) and not a plain dict equal

        # Test Module 2
        module2_id = vessel_dict["modules"][1][REF_MARKER].split(ref_prefix)[1]
        module2_dict = references[module2_id]
        self.assertIn(REF_MARKER, module2_dict["direction"])

        # Test Vector 2
        vector2_id = module2_dict["direction"][REF_MARKER].split(ref_prefix)[1]
        self.assertIn(vector2_id, references)
        vector2_dict = references[vector2_id]
        self.assertEqual(vector2_dict, expected_vector1_dict)


