import unittest
from typing import List, Tuple, Dict, Any
from dessia_common.core import DessiaObject
from dessia_common.excel_reader import ExcelReader
import openpyxl


class Point(DessiaObject):
    def __init__(self, x: float, name: str = ''):
        self.x = x
        super().__init__(name=name)


class Line(DessiaObject):
    def __init__(self, p1: Point, p2: Point, name: str = ""):
        self.p1 = p1
        self.p2 = p2
        self.name = name

        super().__init__(name)


class ShapeCollection(DessiaObject):
    def __init__(self, lines: List[Line], name: str = ''):
        self.lines = lines
        super().__init__(name=name)


class CompositeShape(DessiaObject):
    def __init__(self, shape_collection: ShapeCollection, name: str = ''):
        self.shape_collection = shape_collection
        super().__init__(name=name)

# TODO: fix case if attributes is List, Tuple, Dict of builtin
# class Circle(DessiaObject):
#     def __init__(self, radius: float, center: Tuple[float, float], color: str = '', name: str = ''):
#         self.radius = radius
#         self.center = center
#         self.color = color
#         super().__init__(name=name)
#
#
# class Polygon(DessiaObject):
#     def __init__(self, vertices: List[Tuple[float, float]], properties: Dict[str, Any], circles: List[Circle],
#                  name: str = ''):
#         self.vertices = vertices
#         self.properties = properties
#         self.circles = circles
#         super().__init__(name=name)


class TestReadExcel(unittest.TestCase):
    def test_container(self):

        point1 = Point(0, 'A')
        point2 = Point(1, 'B')
        point3 = Point(2, 'C')
        point4 = Point(3, 'D')

        line1 = Line(point1, point2, 'AB')
        line2 = Line(point2, point3, 'BC')
        line3 = Line(point3, point4, 'CD')

        lines_collection = ShapeCollection([line1, line2, line3], 'Shape Collection')
        composite_shape = CompositeShape(lines_collection, 'Composite Shape')

        excel_file_path = "composite_shape.xlsx"
        composite_shape.to_xlsx(excel_file_path)

        reader = ExcelReader(excel_file_path)

        cell_values = reader.process_workbook()
        workbook = openpyxl.load_workbook(filename=excel_file_path)
        self.assertEqual(len(cell_values.keys()), len(workbook.worksheets))

        main_obj = reader.read_object()
        self.assertEqual(main_obj, composite_shape)



