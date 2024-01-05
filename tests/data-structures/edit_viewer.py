"""

"""

from typing import List
from dessia_common.core import DessiaObject
from dessia_common.decorators import plot_data_view

import volmdlr as vm
import volmdlr.primitives2d as p2d

import plot_data
from plot_data.colors import RED, BLUE, WHITE


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

    def contour(self, reference_path: str = "#"):
        points = [vm.Point2D(0, 0), vm.Point2D(0, self.width),
                  vm.Point2D(self.length, self.width), vm.Point2D(self.length, 0)]
        return p2d.ClosedRoundedLineSegments2D(points=points, radius={}, reference_path=reference_path)

    @plot_data_view("2D View")
    def plot2d(self, reference_path: str = "#"):
        contour = self.contour(reference_path)
        edge_style = plot_data.EdgeStyle(color_stroke=RED)
        fill_style = plot_data.SurfaceStyle(color_fill=WHITE)
        return contour.plot_data(edge_style=edge_style, surface_style=fill_style)


class VerticalBeam(Beam):
    _standalone_in_db = True

    def __init__(self, length: float, name: str = ""):
        super().__init__(length=length, name=name)

    def contour(self, origin: float, reference_path: str = "#"):
        points = [vm.Point2D(origin, 0), vm.Point2D(origin, self.length),
                  vm.Point2D(origin + self.width, self.length), vm.Point2D(origin + self.width, 0)]
        return p2d.ClosedRoundedLineSegments2D(points=points, radius={}, reference_path=reference_path)

    @plot_data_view("2D View")
    def plot2d(self, origin: float, reference_path: str = "#"):
        contour = self.contour(origin=origin, reference_path=reference_path)
        edge_style = plot_data.EdgeStyle(color_stroke=BLUE)
        fill_style = plot_data.SurfaceStyle(color_fill=WHITE)
        return contour.plot_data(edge_style=edge_style, surface_style=fill_style)


class BeamStructure(DessiaObject):
    _standalone_in_db = True

    def __init__(self, horizontal_beam: HorizontalBeam, vertical_beams: List[VerticalBeam], name: str = ""):
        self.horizontal_beam = horizontal_beam
        self.vertical_beams = vertical_beams

        super().__init__(name=name)

    @plot_data_view("2D View")
    def plot2d(self, reference_path: str = "#"):
        horizontal_contour = self.horizontal_beam.plot2d(reference_path=f"{reference_path}/horizontal_beam")
        vertical_contours = [b.plot2d(origin=self.horizontal_beam.length * i / len(self.vertical_beams),
                                      reference_path=f"{reference_path}/vertical_beams/{i}")
                             for i, b in enumerate(self.vertical_beams)]
        labels = [plot_data.Label(c.reference_path, shape=c) for c in [horizontal_contour] + vertical_contours]
        primtives = [horizontal_contour] + vertical_contours + labels
        return plot_data.PrimitiveGroup(primitives=primtives, name="Contour")


horizontal = HorizontalBeam(10, "H")
verticals = [VerticalBeam(5, "V1"), VerticalBeam(10, "V2"), VerticalBeam(7.5, "V3")]
structure = BeamStructure(horizontal_beam=horizontal, vertical_beams=verticals, name="Structure")
