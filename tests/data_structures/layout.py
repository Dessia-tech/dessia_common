from typing import List
from dessia_common.core import DessiaObject


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
