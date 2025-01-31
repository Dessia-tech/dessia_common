import unittest
from dessia_common import REF_MARKER

from typing import List
from dessia_common.core import DessiaObject


class Point:
    def __init__(self, x: float):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

    def __hash__(self):
        return self.x

    def to_dict(self):
        return {"x": self.x}

    @classmethod
    def dict_to_object(cls, dict_):
        return cls(x=dict_["x"])


class Container(DessiaObject):
    def __init__(self, points: List[Point], name: str = ""):
        self.points = points
        self.name = name

        super().__init__(name)


class Line(DessiaObject):
    def __init__(self, p1: Point, p2: Point, name: str = ""):
        self.p1 = p1
        self.p2 = p2
        self.name = name

        super().__init__(name)


a = Point(0)
b = Point(1)
c = Point(0)


class TestNonDessiaObjects(unittest.TestCase):
    def setUp(self):
        self.ref_prefix = "#/_references/"

    def test_container_with_equality(self):
        container = Container([a, b, c])
        dict_ = container.to_dict()
        self.assertIn(REF_MARKER, dict_["points"][0])
        self.assertIn("#/_references/", dict_["points"][0][REF_MARKER])
        point_id = dict_["points"][0][REF_MARKER].split(self.ref_prefix)[1]
        self.assertEqual(dict_["points"][1], container.points[1].to_dict())
        self.assertEqual(dict_["points"][2], {REF_MARKER: f"{self.ref_prefix}{point_id}"})

    def test_line_with_equality(self):
        line = Line(a, c)
        dict_ = line.to_dict()
        self.assertIn(REF_MARKER, dict_["p1"])
        self.assertIn("#/_references/", dict_["p1"][REF_MARKER])
        point_id = dict_["p1"][REF_MARKER].split(self.ref_prefix)[1]
        self.assertEqual(dict_["p2"], {REF_MARKER: f"{self.ref_prefix}{point_id}"})

    def test_container_without_equality(self):
        container = Container([a, b])
        dict_ = container.to_dict()
        self.assertDictEqual(dict_["_references"], {})
        for i, point in enumerate(container.points):
            self.assertEqual(dict_["points"][i], point.to_dict())

    def test_line_without_equality(self):
        line = Line(a, b)
        dict_ = line.to_dict()
        self.assertDictEqual(dict_["_references"], {})
        self.assertEqual(dict_["p1"], line.p1.to_dict())
        self.assertEqual(dict_["p2"], line.p2.to_dict())
