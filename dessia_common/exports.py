#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# import collections
import tempfile
# import numpy as npy

# import dessia_common.core
from dessia_common.breakdown import object_breakdown


# from openpyxl.writer.excel import save_virtual_workbook
from openpyxl.styles.borders import Border, Side
from openpyxl.styles import Alignment, PatternFill, Font
from openpyxl import Workbook
import openpyxl.utils


def is_hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def is_number(v):
    return isinstance(v, int) or isinstance(v, float)


def is_builtins_list(l):
    for e in l:
        if not (is_number(e) or isinstance(e, str)):
            return False
    return True


class XLSXWriter:

    max_column_width = 40
    color_dessIA1 = "263238"
    color_dessIA2 = "537CB0"
    grey1 = "f1f1f1"
    white_font = Font(color="FFFFFF")

    thin_border = Border(left=Side(style='thin'),
                         right=Side(style='thin'),
                         top=Side(style='thin'),
                         bottom=Side(style='thin'))

    def __init__(self, object_):

        self.pattern_color1 = PatternFill(
            fill_type="solid",
            start_color=self.color_dessIA1,
            end_color=self.color_dessIA1)

        self.pattern_color2 = PatternFill(
            fill_type="solid",
            start_color=self.color_dessIA2,
            end_color=self.color_dessIA2)

        self.workbook = Workbook()
        self.main_sheet = self.workbook.active
        self.object = object_

        self.paths = object_breakdown(object_)

        self.classes_to_sheets = {}
        self.object_to_sheet_row = {}
        for class_name, obj_paths in self.paths.items():
            sheet = self.workbook.create_sheet(class_name)
            self.classes_to_sheets[class_name] = sheet

            for i, (obj, path) in enumerate(obj_paths.items()):
                self.object_to_sheet_row[obj] = (sheet, i + 2, path)

        self.write()

    def write_class_header_to_row(self, obj_of_class, sheet, row_number, path=''):
        cell = sheet.cell(row=row_number, column=1, value='Path')
        cell.fill = self.pattern_color2
        cell.border = self.thin_border
        cell.font = self.white_font

        cell = sheet.cell(row=row_number, column=2, value='name')
        cell.fill = self.pattern_color2
        cell.border = self.thin_border
        cell.font = self.white_font
        i = 3

        for (k, _) in sorted(obj_of_class.__dict__.items()):
            if (not k.startswith('_')) and k != 'name':
                cell = sheet.cell(row=row_number, column=i, value=str(k))

                cell.border = self.thin_border
                cell.fill = self.pattern_color2
                cell.font = self.white_font
                i += 1

    def write_object_to_row(self, obj, sheet, row_number, path=''):
        cell = sheet.cell(row=row_number, column=1, value=path)
        cell.border = self.thin_border
        if hasattr(obj, 'name'):
            cell = sheet.cell(row=row_number, column=2, value=obj.name)
        else:
            cell = sheet.cell(row=row_number, column=2, value='No name in model')
        cell.border = self.thin_border
        i = 3
        for (k, v) in sorted(obj.__dict__.items()):
            if (not k.startswith('_')) and k != 'name':
                cell_link = None
                if isinstance(v, dict):
                    str_v = 'Dict of {} items'.format(len(v))
                elif isinstance(v, list):
                    if is_builtins_list(v):
                        str_v = str(v)
                    else:
                        str_v = 'List of {} items'.format(len(v))

                elif isinstance(v, set):
                    str_v = 'Set of {} items'.format(len(v))
                elif isinstance(v, float):
                    str_v = round(v, 6)
                elif is_hashable(v) and v in self.object_to_sheet_row:
                    ref_sheet, ref_row_number, ref_path = self.object_to_sheet_row[v]
                    str_v = ref_path
                    cell_link = '#{}!A{}'.format(ref_sheet.title, ref_row_number)
                else:
                    str_v = str(v)

                cell = sheet.cell(row=row_number, column=i, value=str_v)
                if cell_link:
                    cell.hyperlink = cell_link

                cell.border = self.thin_border

                i += 1

                # column_width = min((len(k) + 1.5), max_column_width)
                # column_name = openpyxl.utils.cell.get_column_letter(i)
                # sheet.column_dimensions[column_name].width = column_width

    def write_object_id(self, sheet):
        sheet.title = 'Object {}'.format(self.object.__class__.__name__)

        sheet['A1'] = 'Module'
        sheet['B1'] = 'Class'
        sheet['C1'] = 'name'

        sheet['A1'].border = self.thin_border
        sheet['B1'].border = self.thin_border
        sheet['C1'].border = self.thin_border
        sheet['A1'].fill = self.pattern_color1
        sheet['B1'].fill = self.pattern_color1
        sheet['C1'].fill = self.pattern_color1
        sheet['A1'].font = self.white_font
        sheet['B1'].font = self.white_font
        sheet['C1'].font = self.white_font

        sheet['A2'] = self.object.__module__
        sheet['B2'] = self.object.__class__.__name__
        sheet['C2'] = self.object.name

        sheet['A2'].border = self.thin_border
        sheet['B2'].border = self.thin_border
        sheet['C2'].border = self.thin_border
        sheet['A2'].fill = self.pattern_color1
        sheet['B2'].fill = self.pattern_color1
        sheet['C2'].fill = self.pattern_color1
        sheet['A2'].font = self.white_font
        sheet['B2'].font = self.white_font
        sheet['C2'].font = self.white_font

        sheet['A3'] = 'Attribute'
        sheet['A4'] = 'Value'
        sheet['A3'].border = self.thin_border
        sheet['A4'].border = self.thin_border

    def write(self):
        # name_column_width = 0
        self.write_object_id(self.main_sheet)
        self.write_class_header_to_row(self.object, self.main_sheet, 3)
        self.write_object_to_row(self.object, self.main_sheet, 4)
        self.autosize_sheet_columns(self.main_sheet, 5, 30)

        for class_name, obj_paths in self.paths.items():
            sheet = self.classes_to_sheets[class_name]

            for obj, path in obj_paths.items():
                _, row_number, path = self.object_to_sheet_row[obj]
                self.write_object_to_row(obj, sheet, row_number, path)
            self.write_class_header_to_row(obj, sheet, 1)

            sheet.auto_filter.ref = "A1:{}{}".format(openpyxl.utils.cell.get_column_letter(sheet.max_column),
                                                     len(obj_paths) + 1)
            self.autosize_sheet_columns(sheet, 5, 30)

    def save_to_file(self, filepath):

        if isinstance(filepath, str):
            real_filepath = filepath
            if not filepath.endswith('.xlsx'):
                real_filepath += '.xlsx'
        else:
            real_filepath = tempfile.NamedTemporaryFile().name

        self.workbook.save(real_filepath)

        if not isinstance(filepath, str):
            with open(real_filepath, 'rb') as file:
                filepath.seek(0)
                filepath.write(file.read())

    def autosize_sheet_columns(self, sheet, min_width=5, max_width=30):
        # Autosize columns
        for col in sheet.columns:
            width = min_width
            column = col[1].column_letter  # Get the column name
            # Since Openpyxl 2.6, the column name is  ".column_letter" as .column became the column number (1-based)
            for cell in col:
                try:  # Necessary to avoid error on empty cells
                    if len(str(cell.value)) > width:
                        width = len(cell.value)
                except:
                    pass
            if width > 0:
                adjusted_width = min((width + 0.5), max_width)
                sheet.column_dimensions[column].width = adjusted_width
