#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 2023

@author: gvouaillat
"""

import io
import math
import random
from typing import List

import numpy as npy
import scipy as scp

import plot_data.core as plot_data
from dessia_common.core import DessiaObject, PhysicalObject
from dessia_common.files import BinaryFile
from plot_data import colors
from volmdlr import wires, geometry, edges, Point2D, faces, step


def check_positif(dimension):
    """ Check if the dimension is acceptable (if positif) or not. """
    if dimension <= 0:
        print('Please Enter a Valid (positif) Value')
        raise ValueError('Please Enter a Valid (positif) Value')


def rotation_angle(stringer_line, principle_axe):
    """
    Finds out the rotation angle between the stringer line and the section principle axe.

    :param stringer_line: A stringer
    :type stringer_line: edges.Line2D
    :param principle_axe: The principle axe
    :type principle_axe: edges.Line2D

    :return: A rotation angle
    :rtype: float
    """
    def minimize_angle(x_param: float, principle_axe_: edges.Line2D, center_: Point2D, stringer_line_: edges.Line2D):
        principle_axe_new = principle_axe_.rotation(center_, x_param)

        if stringer_line_.unit_direction_vector().to_vector().y < 0:
            return geometry.clockwise_angle(principle_axe_new.unit_direction_vector().to_vector(),
                                            -stringer_line_.unit_direction_vector().to_vector())
        return geometry.clockwise_angle(principle_axe_new.unit_direction_vector().to_vector(),
                                        stringer_line_.unit_direction_vector().to_vector())

    error = math.inf
    if stringer_line.unit_direction_vector().to_vector().y < 0:
        angle = geometry.clockwise_angle(principle_axe.unit_direction_vector().to_vector(),
                                         -stringer_line.unit_direction_vector().to_vector())
    else:
        angle = geometry.clockwise_angle(principle_axe.unit_direction_vector().to_vector(),
                                         stringer_line.unit_direction_vector().to_vector())

    x_init = list(npy.arange(-angle, angle, angle / 20))

    center = stringer_line.line_intersections(principle_axe)[0]

    for x_0 in x_init:

        sol = scp.optimize.minimize(minimize_angle, x0=x_0,
                                    args=(principle_axe, center, stringer_line),
                                    options={'eps': 1e-12})
        if sol.fun < error:
            solution = sol.x
            error = sol.fun
    return solution


def data2d_rotation_angle(frame: wires.Contour2D):
    """
    Finds out the rotation angle to best position the frame in 2D plane.

    :param frame: A frame contour
    :type frame: vm.wires.Contour2D

    :return: A rotation angle
    :rtype: float
    """

    def minimize_bounding_box(x_param: float, frame_: wires.Contour2D):
        """ Minimizes the bounding box of frame2d. """

        center = frame_.center_of_mass()
        new_frame = frame_.rotation(center, x_param)
        xmini, xmaxi, ymini, ymaxi = new_frame.bounding_rectangle
        bounding_box = wires.Contour2D.from_bounding_rectangle(xmini, xmaxi, ymini, ymaxi)
        return abs(new_frame.area() - bounding_box.area())

    x_init = random.sample(range(-int(2 * math.pi), int(2 * math.pi)), 5)
    error = 2

    for x_0 in x_init:
        sol = scp.optimize.minimize(minimize_bounding_box, x0=x_0, args=frame)
        if sol.fun < error:
            error = sol.fun
            solution = sol.x

    frame_new = frame.rotation(center=frame.center_of_mass(), angle=solution)

    to_break = False
    for angle in [math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]:
        frame_new = frame_new.rotation(center=frame_new.center_of_mass(), angle=math.pi / 2)
        _, _, ymin, _ = frame_new.bounding_rectangle
        for prim in frame_new.primitives:
            # TODO: Check abs_tol=0.1, when the Bug #699 volmdlr is fixed
            if math.isclose(prim.middle_point().y, ymin, abs_tol=0.1):
                to_break = True
                break
        if to_break:
            break

    if to_break:
        return angle + solution
    return solution


def plotdata_elements_contours(elements: dict):
    """
    Plots elements (frames/stringers), using plot_data library

    :param elements: A group of wires2D
    :type elements: dict
    :return: A group of plot_data primitives
    :rtype: List[plot_data.LineSegment2D]
    """

    edges_style = {'section': plot_data.EdgeStyle(line_width=1, color_stroke=colors.BLACK, dashline=[]),
                   'frame': plot_data.EdgeStyle(line_width=2, color_stroke=colors.BLUE, dashline=[]),
                   'stringers': plot_data.EdgeStyle(line_width=2, color_stroke=colors.GREEN, dashline=[]),
                   'pockets': plot_data.EdgeStyle(line_width=3, color_stroke=colors.RED, dashline=[])}

    contours = []
    for key, element in elements.items():
        for contour in element:
            primitives = []
            for primitive in contour.primitives:
                if isinstance(primitive, edges.LineSegment2D):
                    primitives.append(plot_data.LineSegment2D(
                        [primitive.start.x, primitive.start.y],
                        [primitive.end.x, primitive.end.y],
                        edge_style=edges_style[key]))
                elif isinstance(primitive, (edges.Arc2D,
                                            edges.BSplineCurve2D)):
                    points = primitive.discretization_points(number_points=50)
                    for k in range(0, len(points) - 1):
                        primitives.append(plot_data.LineSegment2D(
                            [points[k].x, points[k].y],
                            [points[k + 1].x, points[k + 1].y],
                            edge_style=edges_style[key]))
            contours.append(plot_data.Contour2D(plot_data_primitives=primitives, edge_style=edges_style[key]))
    return contours


class SectionZDimensions(DessiaObject):
    """
    A class that defines the semelle/talon/middle segments_parameters for Z section (length, thickness).

    :param length: The Length
    :param thickness: The thickness
    """

    def __init__(self, length: float, thickness: float, name: str = ''):
        self.length = length
        self.thickness = thickness
        check_positif(length)
        check_positif(thickness)
        DessiaObject.__init__(self, name=name)


class SectionZRadius(DessiaObject):
    """
    A class that defines interior and exterior radius for Z section.

    :param interior_radius: The interior radius
    :param exterior_radius: The exterior radius
    """

    def __init__(self, interior_radius: float, exterior_radius: float, name: str = ''):
        self.interior_radius = interior_radius
        self.exterior_radius = exterior_radius
        check_positif(interior_radius)
        check_positif(exterior_radius)
        DessiaObject.__init__(self, name=name)


class SectionZSegments(DessiaObject):
    """
    A class that defines the semelle/talon/middle parameters for Z section using SectionZDimensions.

    :param semelle: The semelle parameters for Z section
    :param talon: The talon parameters for Z section
    :param middle: The middle parameters for Z section
    """

    def __init__(self, semelle: SectionZDimensions, talon: SectionZDimensions,
                 middle: SectionZDimensions, name: str = ''):
        self.semelle = semelle
        self.talon = talon
        self.middle = middle
        DessiaObject.__init__(self, name=name)


class Section(DessiaObject):
    """ A parent class for all section types. """

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)


class SectionZ(Section):
    """
    A class that defines the Z stringer section.

    :param semelle: The semelle parameters (thickness, length)
    :param talon: The talon parameters (thickness, length)
    :param middle: The middle parameters (thickness, length)
    :param radius: The radius parameters (interior, exterior)
    :param contour: The outer contour of the SectionZ
    :param principle_axis: The middle axe
    """

    _standalone_in_db = True

    def __init__(self, semelle: SectionZDimensions, talon: SectionZDimensions, middle: SectionZDimensions,
                 radius: SectionZRadius, contour: wires.Contour2D = None, principle_axis: edges.Line2D = None,
                 name: str = ''):
        self.semelle = semelle
        self.talon = talon
        self.middle = middle
        self.radius = radius
        self.color = 'k'
        self.contour = contour
        self.principle_axis = principle_axis
        Section.__init__(self, name=name)

    def define_contour(self):
        """
        Compute the outer contour of the Z stringer section.

        :return: The outer contour
        :rtype: wires.Contour2D
        """
        arc_exterior_1 = edges.Arc2D(
            start=Point2D(self.thickness + self.exterior_radius, self.semelle_thickness),
            interior=Point2D(self.thickness + self.exterior_radius - self.exterior_radius / math.sqrt(2),
                             self.semelle_thickness + self.exterior_radius - self.exterior_radius / math.sqrt(2)),
            end=Point2D(self.thickness, self.semelle_thickness + self.exterior_radius)
        )
        arc_interior_1 = edges.Arc2D(
            start=Point2D(self.thickness, self.length - self.interior_radius),
            interior=Point2D(self.thickness - (self.interior_radius - self.interior_radius / math.sqrt(2)),
                             self.length - self.interior_radius + self.interior_radius / math.sqrt(2)),
            end=Point2D(self.thickness - self.interior_radius, self.length)
        )
        arc_exterior_2 = edges.Arc2D(start=Point2D(-self.exterior_radius, self.length - self.talon_thickness),
                                     interior=Point2D(-self.exterior_radius + self.exterior_radius / math.sqrt(2),
                                                      self.length - self.talon_thickness
                                                      - (self.exterior_radius - self.exterior_radius / math.sqrt(2))),
                                     end=Point2D(0, self.length - self.talon_thickness - self.exterior_radius))
        arc_interior_2 = edges.Arc2D(start=Point2D(0, self.interior_radius),
                                     interior=Point2D(self.interior_radius - self.interior_radius / math.sqrt(2),
                                                      self.interior_radius - self.interior_radius / math.sqrt(2)),
                                     end=Point2D(self.interior_radius, 0))
        primitives = [edges.LineSegment2D(Point2D(self.interior_radius, 0), Point2D(self.semelle_length, 0)),
                      edges.LineSegment2D(Point2D(self.semelle_length, 0),
                                          Point2D(self.semelle_length, self.semelle_thickness)),
                      edges.LineSegment2D(Point2D(self.semelle_length, self.semelle_thickness),
                                          Point2D(self.thickness + self.exterior_radius, self.semelle_thickness)),
                      arc_exterior_1,
                      edges.LineSegment2D(Point2D(self.thickness, self.semelle_thickness + self.exterior_radius),
                                          Point2D(self.thickness, self.length - self.interior_radius)),
                      arc_interior_1,
                      edges.LineSegment2D(Point2D(self.thickness - self.interior_radius, self.length),
                                          Point2D(self.thickness - self.talon_length, self.length)),
                      edges.LineSegment2D(Point2D(self.thickness - self.talon_length, self.length),
                                          Point2D(self.thickness - self.talon_length,
                                                  self.length - self.talon_thickness)),
                      edges.LineSegment2D(Point2D(self.thickness - self.talon_length,
                                                  self.length - self.talon_thickness),
                                          Point2D(-self.exterior_radius, self.length - self.talon_thickness)),
                      arc_exterior_2,
                      edges.LineSegment2D(Point2D(0, self.length - self.talon_thickness - self.exterior_radius),
                                          Point2D(0, self.interior_radius)),
                      arc_interior_2]
        return wires.Contour2D(primitives)

    @property
    def exterior_radius(self):
        """
        The  exterior radius.

        :return: The radius
        :rtype: float
        """
        return self.radius.exterior_radius

    @property
    def interior_radius(self):
        """
        The interior radius.

        :return: The radius
        :rtype: float
        """
        return self.radius.interior_radius

    @property
    def length(self):
        """
        The total length.

        :return: The length
        :rtype: float
        """
        return self.middle.length

    @property
    def origin(self):
        """
        Gets the origin.

        :return: A point2d
        :rtype: vm.Point2D
        """
        if not self.principle_axis:
            principle_axis = self.get_principle_axis()
            self.principle_axis = principle_axis
        return self.principle_axis.line_intersections(self.contour.primitives[0].to_line())[0]

    def get_principle_axis(self):
        """
        Gets the principle axe of the Z section.

        :return: An axe
        :rtype: edges.Line2D
        """
        return self.get_principle_primitives()[0].translation(Point2D(-self.thickness / 2, 0)).to_line()

    def get_principle_primitives(self):
        """
        Gets the principle primitives.

        :return: A list of primitives
        :rtype: List
        """
        if not self.contour:
            contour = self.define_contour()
            self.contour = contour

        contour = self.contour
        return contour.primitives[4], contour.primitives[10]

    @property
    def semelle_length(self):
        """
        The semelle length.

        :return: The length
        :rtype: float
        """
        return self.semelle.length

    @property
    def semelle_thickness(self):
        """
        The semelle thickness (vertical).

        :return: The thickness
        :rtype: float
        """
        return self.semelle.thickness

    @property
    def talon_length(self):
        """
        The talon length.

        :return: The length
        :rtype: float
        """
        return self.talon.length

    @property
    def talon_thickness(self):
        """
        The talon thickness (vertical).

        :return: The thickness
        :rtype: float
        """
        return self.talon.thickness

    @property
    def thickness(self):
        """
        The middle thickness (horizontal).

        :return: The thickness
        :rtype: float
        """
        return self.middle.thickness


class Data2D(PhysicalObject):
    """
    A class that represents the 2D image of Data3D.
    It is defined with a contour2d and a list of lines (frame and stringers).

    :param frame2d: A frame
    :param stringers2d: A list of stringers
    """

    _standalone_in_db = True

    def __init__(self, frame2d: wires.Contour2D, stringers2d: List[wires.Wire2D], name: str = ''):
        self.frame2d = frame2d
        self.stringers2d = stringers2d
        PhysicalObject.__init__(self, name=name)

    def position_section(self, section, stringer_index):
        """
        Position an initial section based on it corresponding stringer.

        :param section: An initial section
        :type section: sections.Section
        :param stringer_index: An index
        :type stringer_index: int

        :return: A new section
        :rtype: sections.Section
        """
        section_new = self.position_section_rotation_initial(section, stringer_index)
        intersections = self.frame_stringer_intersection(stringer_index)
        section_new = self.position_section_displacement(section_new, intersections[0])
        return section_new

    def position_sections(self, sections_initial):
        """
        Position a list of initial sections based on its corresponding stringers.

        :param sections_initial: A list of initial sections
        :type sections_initial: List[sections.Section]

        :return: A list of new sections
        :rtype: sections.Section
        """
        return [self.position_section(section, index) for index, section in enumerate(sections_initial)]

    def frame_stringer_intersection(self, stringer_index):
        """
        Finds out the intersection point between the frame and the choosen stringer.

        :param stringer_index: An index
        :type stringer_index: int

        :return: A point2d
        :rtype: vm.Point2D
        """

        stringer = self.stringers2d[stringer_index].primitives[0].to_line()
        intersections = self.frame2d.line_intersections(stringer)
        xmin, xmax, ymin, ymax = self.frame2d.bounding_rectangle.bounds()
        bounding_contour = wires.Contour2D.from_bounding_rectangle(xmin, xmax, ymin, ymax)
        bounding_intersection_points = bounding_contour.line_intersections(stringer)
        for (point, edge) in bounding_intersection_points:
            if edge != bounding_contour.primitives[2]:
                point_intersection_bounding = point
        sorted(intersections, key=lambda element: point_intersection_bounding.point_distance(element[0]))
        return intersections[0]

    def position_section_displacement(self, section, point_intersection):
        """
        Moves the section so as the section touchs the frame contour.

        :param section: A section
        :type section: sections.Section
        :param point_intersection: A point2d
        :type point_intersection: vm.Point2D

        :return: A new section
        :rtype: sections.Section
        """

        if not section.contour:
            contour = section.define_contour()
            section.contour = contour
        if not section.principle_axe:
            principle_axe = section.get_principle_axe()
            section.principle_axe = principle_axe
        contour = section.contour.translation((point_intersection - section.origin).to_vector())
        section_new = section.copy()
        section_new.contour = contour
        section_new.principle_axe = section.principle_axe.translation((point_intersection - section.origin).to_vector())
        return section_new

    def position_section_rotation_initial(self, section, stringer_index):
        """
        Rotates a section based on the data2d corresponding stringer.

        :param section: A section
        :type section: sections.Section
        :param stringer_index: An index
        :type stringer_index: int

        :return: A rotated section
        :rtype: sections.Section
        """

        if not section.contour:
            contour = section.define_contour()
            section.contour = contour

        if not section.principle_axis:
            principle_axe = section.get_principle_axis()
            section.principle_axe = principle_axe

        principle_axe = section.principle_axe
        stringer_line = self.stringers2d[stringer_index].primitives[0].to_line()
        if stringer_line.direction_vector().is_colinear_to(principle_axe.direction_vector()):
            return section

        point_intersection = stringer_line.line_intersections(principle_axe)[0]
        angle = rotation_angle(stringer_line, principle_axe)

        contour = section.contour.rotation(point_intersection, angle)
        section_new = section.copy()
        section_new.contour = contour
        section_new.principle_axe = section.principle_axe.rotation(point_intersection, angle)
        return section_new

    def rotation(self, angle, center=None):
        """
        Rotates data2d.

        :param angle: An angle
        :type angle: float
        :param center: A center, defaults to None
        :type center: vm.Point2D, optional

        :return: A rotated data2d
        :rtype: Data2D
        """
        if center is None:
            center = self.frame2d.center_of_mass()
        frame2d = self.frame2d.rotation(angle=angle, center=center)
        stringers2d = [stringer.rotation(angle=angle, center=center) for stringer in self.stringers2d]
        return self.__class__(frame2d, stringers2d)

    @classmethod
    def from_data3d(cls, data3d: 'Data3D'):
        """
        Defines a Data2D from a Data3D.

        :param data3d: A class that groups elements3d
        :type data3d: Data3D
        :return: A class that groups the 2D image of elements3d
        :rtype: Data2D
        """

        frame2d = data3d.frame3d.to_2d()
        stringers2d = [stringer.to_2d(data3d.frame3d) for stringer in data3d.stringers3d]
        data2d = cls(frame2d, stringers2d)
        angle = data2d_rotation_angle(frame2d)
        return data2d.rotation(center=frame2d.center_of_mass(), angle=angle)


class Data3D(PhysicalObject):
    """
    A class that groups together the 3D elements used for
    the study (frames and stringers).

    :param frame3d: A frame
    :param stringers3d: A list of stringers
    """

    _standalone_in_db = True

    def __init__(self, frame3d: 'Frame', stringers3d: List['Stringer'], name: str = ''):
        self.frame3d = frame3d
        self.stringers3d = stringers3d
        PhysicalObject.__init__(self, name=name)


class DataForPockets(DessiaObject):
    """
    A class that regroups Data2D and positionated stringer sections.

    :param data2d: The 2D image of Data3D
    :param sections_list: A list of sections (with good positions related to stringers)
    """

    _standalone_in_db = True

    def __init__(self, data2d: Data2D, sections_list: List[SectionZ], name: str = ''):
        self.data2d = data2d
        self.sections_list = sections_list
        DessiaObject.__init__(self, name=name)

    @classmethod
    def from_data2d_and_initial_sections(cls, data2d: Data2D, sections_initial: List[SectionZ]):
        """
        Define data_for_pockets from data2d and a list of initial sections.

        :param data2d: A data2d
        :param sections_initial: A list of initial sections
        """
        nsections = len(sections_initial)
        nstringers = len(data2d.stringers2d)
        if nsections != nstringers:
            raise ValueError(f"There are not as much sectionz {nsections} as stringers {nstringers}")
        new_sections = data2d.position_sections(sections_initial)
        return cls(data2d, new_sections)

    def plot_data(self, reference_path: str = "#", **kwargs):
        """
        Plots the DataForPockets using plot_data library.
        """

        primitives = plotdata_elements_contours({'frame': [self.data2d.frame2d], 'stringers': self.data2d.stringers2d})
        sections_contours = [section.contour for section in self.sections_list]
        primitives.extend(plotdata_elements_contours({'section': sections_contours}))
        primitive_group = plot_data.PrimitiveGroup(primitives=primitives)
        return [primitive_group]


class Element3D(PhysicalObject):
    """
    A class that defines the orbital/longitudinal
    structures in 3D (Frame & Stringer).

    :param face: A face3d defined from a STEP file
    """

    _standalone_in_db = True

    def __init__(self, face: faces.Face3D, name: str = ''):
        self.face = face
        PhysicalObject.__init__(self, name=name)

    def element3d_intersection(self, element3d):
        """
        Returns the intersection between two elements3d (a frame with a stringer).

        :param element3d: A stringer or a frame
        :type element3d: Element3D

        :return: A wire
        :rtype: wires.Wire3D
        """
        return self.face.face_intersections(element3d.face)


class Frame(Element3D):
    """
    A class that defines the Frame with a face3d, inherited from
    Element3D class.

    :param face: A face3d defined from a STEP file
    """

    _standalone_in_db = True

    def __init__(self, face: faces.Face3D, name: str = ''):
        self.face = face
        self.contour = self.face.outer_contour3d
        self.color = 'b'
        Element3D.__init__(self, face=face, name=name)

    def to_2d(self):
        """
        Projects the frame on 2d plane.

        :return: The 2D image of a frame
        :rtype: wires.Contour2D
        """
        return self.face.surface2d.outer_contour

    @classmethod
    def from_input_data(cls, input_data: 'InputData'):
        """
        Uses the faces extracted from the STEP file to define a Frame.

        :param input_data: Geometrical input data (frame)
        :type input_data: InputData

        :return: A frame3d
        :rtype: Frame
        """
        return cls(input_data.list_faces[0])


class Stringer(Element3D):
    """
    A class that defines the Stringer with a face3d, inherited from
    Element3D class.

    :param face: A face3d defined from a STEP file
    :type face faces.Face3D
    """

    _standalone_in_db = True

    def __init__(self, face: faces.Face3D, name: str = ''):
        """
        Constructor of class Stringer
        """

        self.face = face
        self.color = 'g'
        Element3D.__init__(self, face=face, name=name)

    def to_2d(self, frame: Frame):
        """
        Projects the stringer on 2D based on the frame plane.

        :param frame: A frame

        :return: A wire (contour)
        :rtype: wires.Contour2D
        """

        return frame.face.surface3d.contour3d_to_2d(wires.Contour3D(self.to_wire(frame).primitives))

    def to_wire(self, frame: Frame):
        """
        Converts the stringer to a line.
        This line is the intersection result with a frame.

        :param frame: A frame

        :return: An intersection wire
        :rtype: wires.Wire3D
        """
        return self.element3d_intersection(frame)[0]

    @classmethod
    def from_input_data(cls, input_data: 'InputData'):
        """
        Uses the faces extracted from the STEP file to define a list
        of stringers.

        :param input_data: Geometrical input data (stringers)

        :return: A list of stringers3d
        :rtype: List[Stringer]
        """
        return [cls(face) for face in input_data.list_faces]


class InputData(DessiaObject):
    """
    A class defined to read STEP files and create 3D geometries.
    It is about frames and stringers.

    :param list_faces: A list of faces3D imported from STEP file
    """

    _standalone_in_db = True

    def __init__(self, list_faces: List[faces.Face3D], name: str = ''):
        self.list_faces = list_faces
        DessiaObject.__init__(self, name=name)

    @classmethod
    def from_step_file(cls, step_file_stream: BinaryFile):
        """
        Reads STEP files of different 3D geometries
        (frames ans stringers) and extract geometrical data (faces3d).

        :param step_file_stream: Stream of the input STEP file

        :return Input geometrical data
        :rtype: InputData
        """
        step_file = step.Step.from_stream(stream=step_file_stream)
        model = step_file.to_volume_model()
        primitives = model.primitives
        list_faces = []
        for primitive in primitives:
            list_faces.extend(primitive.faces)
        return cls(list_faces)


class Intersection(DessiaObject):
    """
    Represents parameters to be set at the intersection of a frame and a stringer
    :param frame: Specific frame for intersection
    :param stringer: Specific stringer for intersection
    :param parameters: List of parameters to be set for intersection characterization
    :param section: Section at the intersection
    :param name: name of the intersection
    """
    _standalone_in_db = True

    def __init__(self, frame: Frame, stringer: Stringer, parameters: List[float], section: SectionZ, name: str = ''):
        """
        Constructor of class PocketParameters
        """
        self.frame = frame
        self.stringer = stringer
        self.parameters = parameters
        self.section = section
        DessiaObject.__init__(self, name=name)

    def do_something_with_parameters(self):
        """
        Does something with parameters
        :return: A printed message
        """
        print('Something complicated has been done with parameters set on the intersection')


# # Existing code
# semelle_dimensions = SectionZDimensions(length=0.02, thickness=0.0012)
# talon_dimensions = SectionZDimensions(length=0.01, thickness=0.0025)
# middle_dimensions = SectionZDimensions(length=0.024, thickness=0.0012)
# radius = SectionZRadius(interior_radius=0.002, exterior_radius=0.0005)
# section_z = SectionZ(semelle=semelle_dimensions, talon=talon_dimensions, middle=middle_dimensions, radius=radius)
#
# file_path = io.FileIO('./models/data/frame_1.step')
# inputdata = InputData.from_step_file(file_path)
# frame = Frame.from_input_data(inputdata)
#
# file_path = io.FileIO('./models/data/stringers_1.step')
# inputdata = InputData.from_step_file(file_path)
# stringers = Stringer.from_input_data(inputdata)
#
# data3d_2 = Data3D(frame, stringers)
# data2d_2 = Data2D.from_data3d(data3d_2)
#
# data_for_pockets_2 = DataForPockets.from_data2d_and_initial_sections(
#     data2d=data2d_2,
#     sections_initial=[section_z] * len(data2d_2.stringers2d)
# )
# data_for_pockets_2.plot()
#
# # Object that we want to create in the form
# # The highlighted object is the corresponding section
# for index, intersect in enumerate(data_for_pockets_2.sections_list):
#     intersection = Intersection(frame=frame, stringer=stringers[index], section=intersect,
#                                 parameters=[0.006, 0.009, 0.004, 0.004, 0.002])
