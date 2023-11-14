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

container = Container([a, b, c])
line = Line(a, c)
