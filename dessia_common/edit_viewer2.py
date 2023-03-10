#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 2023

@author: gvouaillat
"""

from typing import List

import plot_data.core as plot_data
from dessia_common.core import PhysicalObject, DessiaObject
from plot_data import colors
import volmdlr as vm
import volmdlr.primitives2d as p2d
import volmdlr.primitives3d as p3d


class Section(PhysicalObject):
    """ A class that defines the Z stringer section. """

    _standalone_in_db = True

    def __init__(self, size: float, origin: float = 0, name: str = ''):
        self.size = size
        self.origin = origin
        PhysicalObject.__init__(self, name=name)

    def plot_data_contour(self):
        origin = self.origin
        origin_y = 1
        edge_style = plot_data.EdgeStyle(line_width=2, color_stroke=colors.BLACK, dashline=[])
        primitives = [plot_data.LineSegment2D([origin, origin_y - self.size/2],
                                              [origin, origin_y + self.size/2],
                                              edge_style=edge_style)]

        return plot_data.Contour2D(plot_data_primitives=primitives, edge_style=edge_style)

    def plot_data(self, reference_path: str = "#", **kwargs):
        contour = self.plot_data_contour()
        return [plot_data.PrimitiveGroup(primitives=[contour], name='Frame')]

    def volmdlr_primitives(self, **kwargs):
        return [p3d.Sphere(center=vm.Point3D(self.origin, 0, 1), radius=1)]


class Data2D(DessiaObject):
    _standalone_in_db = True

    def __init__(self, frame: 'Frame', stringers: List['Stringer'], name: str = ""):
        self.frame = frame
        self.stringers = stringers
        DessiaObject.__init__(self, name=name)

    def position_section(self, stringer_index):
        """ Position an initial section based on it corresponding stringer. """
        stringer = self.stringers[stringer_index]
        return Section(size=stringer.width / 2, origin=stringer.origin + stringer.width / 2)

    def position_sections(self, sections_initial):
        """ Position a list of initial sections based on its corresponding stringers. """
        return [self.position_section(i) for i, section in enumerate(sections_initial)]


class DataForPockets(PhysicalObject):
    """ A class that regroups Data2D and positionated stringer sections. """

    _standalone_in_db = True

    def __init__(self, data2d: Data2D, sections: List[Section], name: str = ''):
        self.data2d = data2d
        self.sections = sections
        PhysicalObject.__init__(self, name=name)

    @classmethod
    def from_data2d_and_initial_sections(cls, data2d: Data2D, initial_sections: List[Section], name: str = ""):
        """ Define data_for_pockets from data2d and a list of initial sections. """
        nsections = len(initial_sections)
        nstringers = len(data2d.stringers)
        if nsections != nstringers:
            raise ValueError(f"There are not as much sectionz {nsections} as stringers {nstringers}")
        sections = data2d.position_sections(initial_sections)
        return cls(data2d=data2d, sections=sections, name=name)

    def plot_data(self, reference_path: str = "#", **kwargs):
        """ Plots the DataForPockets using plot_data library. """
        primitives = [self.data2d.frame.plot_data_contour()]
        primitives += [s.plot_data_contour() for s in self.data2d.stringers]
        primitives += [s.plot_data_contour() for s in self.sections]
        return [plot_data.PrimitiveGroup(primitives=primitives)]

    def volmdlr_primitives(self, **kwargs):
        primitives = self.data2d.frame.volmdlr_primitives()
        primitives += [s.volmdlr_primitives()[0] for s in self.data2d.stringers]
        primitives += [s.volmdlr_primitives()[0] for s in self.sections]
        return primitives


class Frame(PhysicalObject):
    """ Frame. """

    _standalone_in_db = True

    def __init__(self, length: float = 20, name: str = ''):
        self.length = length
        self.width = 2
        self.thickness = 0.05
        PhysicalObject.__init__(self, name=name)

    def plot_data_contour(self):
        origin = 0
        edge_style = plot_data.EdgeStyle(line_width=2, color_stroke=colors.GREEN, dashline=[])
        primitives = [
            plot_data.LineSegment2D(
                [origin, origin],
                [origin, origin + self.width],
                edge_style=edge_style
            ),
            plot_data.LineSegment2D(
                [origin, origin + self.width],
                [origin + self.length, origin + self.width],
                edge_style=edge_style
            ),
            plot_data.LineSegment2D(
                [origin + self.length, origin + self.width],
                [origin + self.length, origin],
                edge_style=edge_style
            ),
            plot_data.LineSegment2D(
                [origin + self.length, origin],
                [origin, origin],
                edge_style=edge_style
            )
        ]
        return plot_data.Contour2D(plot_data_primitives=primitives, edge_style=edge_style)

    def contour(self):
        """ Squared contour. """
        points = [vm.Point2D(0, 0), vm.Point2D(self.length, 0),
                  vm.Point2D(self.length, self.width), vm.Point2D(0, self.width)]
        crls = p2d.ClosedRoundedLineSegments2D(points=points, radius={})
        return crls

    def volmdlr_primitives(self, **kwargs):
        """ Volmdlr primitives of a cube. """
        contour = self.contour()
        extrusion = p3d.ExtrudedProfile(plane_origin=vm.Point3D(0, 0, 0), x=vm.X3D, y=vm.Z3D,
                                        outer_contour2d=contour, inner_contours2d=[],
                                        extrusion_vector=vm.Y3D * self.thickness)
        return [extrusion]

    def plot_data(self, reference_path: str = "#", **kwargs):
        contour = self.plot_data_contour()
        return [plot_data.PrimitiveGroup(primitives=[contour], name='Frame')]


class Stringer(PhysicalObject):
    """ Stringer. """

    _standalone_in_db = True

    def __init__(self, origin: float = 0., length: float = 10, name: str = ''):
        self.length = length
        self.origin = origin
        self.width = 1
        PhysicalObject.__init__(self, name=name)

    def plot_data_contour(self):
        origin = self.origin
        edge_style = plot_data.EdgeStyle(line_width=2, color_stroke=colors.BLUE, dashline=[])
        primitives = [
            plot_data.LineSegment2D([origin, 0], [origin, self.length], edge_style=edge_style),
            plot_data.LineSegment2D([origin, self.length], [origin + self.width, self.length], edge_style=edge_style),
            plot_data.LineSegment2D([origin + self.width, self.length], [origin + self.width, 0],
                                    edge_style=edge_style),
            plot_data.LineSegment2D([origin + self.width, 0], [origin, 0], edge_style=edge_style)
        ]

        return plot_data.Contour2D(plot_data_primitives=primitives, edge_style=edge_style)

    def plot_data(self, reference_path: str = "#", **kwargs):
        contour = self.plot_data_contour()
        return [plot_data.PrimitiveGroup(primitives=[contour], name='Stringer')]

    @classmethod
    def generate_many(cls, frame: Frame, seed: int):
        space = frame.length // seed
        return [cls(origin=space*i, name=f"Stringer{i}") for i in range(seed)]

    def contour(self):
        """ Squared contour. """
        hwidth = self.width / 2
        points = [vm.Point2D(self.origin, 1 - hwidth), vm.Point2D(self.origin + self.width, 1 - hwidth),
                  vm.Point2D(self.origin + self.width, 1 + hwidth), vm.Point2D(self.origin, 1 + hwidth)]
        crls = p2d.ClosedRoundedLineSegments2D(points=points, radius={})
        return crls

    def volmdlr_primitives(self, **kwargs):
        """ Volmdlr primitives of a cube. """
        contour = self.contour()
        extrusion = p3d.ExtrudedProfile(plane_origin=vm.Point3D(0, 0, 0), x=vm.X3D, y=vm.Z3D,
                                        outer_contour2d=contour, inner_contours2d=[],
                                        extrusion_vector=vm.Y3D * self.length)
        return [extrusion]


class Intersection(PhysicalObject):
    """ Represents parameters to be set at the intersection of a frame and a stringer. """
    _standalone_in_db = True

    def __init__(self, frame: Frame, stringer: Stringer, section: Section, name: str = ''):
        """
        Constructor of class PocketParameters
        """
        self.frame = frame
        self.stringer = stringer
        self.section = section
        PhysicalObject.__init__(self, name=name)

    def plot_data(self, reference_path: str = "#", **kwargs):
        primitives = [self.frame.plot_data_contour(),
                      self.stringer.plot_data_contour(),
                      self.section.plot_data_contour()]
        return plot_data.PrimitiveGroup(primitives=primitives)

    def volmdlr_primitives(self, **kwargs):
        return self.frame.volmdlr_primitives() + self.stringer.volmdlr_primitives() + self.section.volmdlr_primitives()


class Parameters(DessiaObject):

    _standalone_in_db = True

    def __init__(self, data_for_pockets: DataForPockets, parameters: List[float], name: str = ""):
        self.data_for_pockets = data_for_pockets
        self.parameters = parameters

        DessiaObject.__init__(self, name)

    def do_something_with_parameters(self):
        """
        Does something with parameters
        :return: A printed message
        """
        print('Something complicated has been done with parameters set on the intersection')


# Existing code
SECTION = Section(size=0.5, name="Section")
FRAME = Frame(length=20, name="Frame")
STRINGERS = Stringer.generate_many(frame=FRAME, seed=5)

DATA_2D = Data2D(frame=FRAME, stringers=STRINGERS)

DATA_FOR_POCKETS = DataForPockets.from_data2d_and_initial_sections(data2d=DATA_2D,
                                                                   initial_sections=[SECTION] * len(DATA_2D.stringers))

# Object that we want to create in the form
# The highlighted object is the corresponding section
for index, intersect in enumerate(DATA_FOR_POCKETS.sections):
    intersection = Intersection(frame=FRAME, stringer=STRINGERS[index], section=intersect)
