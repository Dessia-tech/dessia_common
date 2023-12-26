"""

"""

from typing import List
from dessia_common.core import DessiaObject
from dessia_common.decorators import plot_data_view

import volmdlr as vm
import volmdlr.primitives2d as p2d

import plot_data


class Beam(DessiaObject):
    _standalone_in_db = True

    def __init__(self, length: float, name: str = ""):
        self.length = length
        self.width = length / 10

        super().__init__(name)


class HorizontalBeam(Beam):
    _standalone_in_db = True

    def __init__(self, length: float, name: str = ""):
        super().__init__(length=length, name=name)

    @plot_data_view("2D View")
    def plot2d(self):
        points = [vm.Point2D(0, 0), vm.Point2D(0, self.width),
                  vm.Point2D(self.length, self.width), vm.Point2D(self.length, 0)]
        return p2d.ClosedRoundedLineSegments2D(points=points, radius={})


class VerticalBeam(Beam):
    _standalone_in_db = True

    def __init__(self, length: float, name: str = ""):
        super().__init__(length=length, name=name)

    @plot_data_view("2D View")
    def plot2d(self, origin: float):
        points = [vm.Point2D(origin, 0), vm.Point2D(origin, self.length),
                  vm.Point2D(origin + self.width, self.length), vm.Point2D(origin + self.width, 0)]
        return p2d.ClosedRoundedLineSegments2D(points=points, radius={})


class BeamStructure(DessiaObject):
    _standalone_in_db = True

    def __init__(self, horizontal_beam: HorizontalBeam, vertical_beams: List[VerticalBeam], name: str = ""):
        self.horizontal_beam = horizontal_beam
        self.vertical_beams = vertical_beams

        super().__init__(name=name)

    @plot_data_view("2D View")
    def plot2d(self):
        horizontal_contour = self.horizontal_beam.plot2d()
        vertical_contours = [b.plot2d(self.horizontal_beam.length * i / len(self.vertical_beams))
                             for i, b in enumerate(self.vertical_beams)]
        return plot_data.PrimitiveGroup(primitives=[horizontal_contour] + vertical_contours, name="Contour")


horizontal = HorizontalBeam(10, "H")
verticals = [VerticalBeam(5, "V1"), VerticalBeam(10, "V2"), VerticalBeam(7.5, "V3")]
structure = BeamStructure(horizontal_beam=horizontal, vertical_beams=verticals, name="Structure")
