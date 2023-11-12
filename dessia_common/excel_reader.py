#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" ExcelReader for DessiaObject. """
import importlib
import inspect

import openpyxl


class ExcelReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.workbook = openpyxl.load_workbook(filepath)
        self.object_class = None
        self.read_objects = {}
        self.main_sheet = self.workbook.worksheets[0].title

    @staticmethod
    def get_location(cell):
        if cell.hyperlink.location:
            target = cell.hyperlink.location.split('!')
        elif cell.hyperlink.target:
            target = cell.hyperlink.target.split('!')
        return target[0].replace("#", ''), target[1]

    def replace_attribute(self, cell, values):
        container = ["Dict", "List", "Set"]
        target = self.get_location(cell)
        dict_keys = []
        if "Dict" in cell.value:
            real_attr_val = {}
            sheet = self.workbook[target]
            column = sheet["A"]
            for row in column[3:]:
                dict_keys.append(row.value.split('.')[1])

            for key, value in zip(dict_keys, values):
                real_attr_val[key] = value

            return real_attr_val

        if isinstance(values, list) and (not any(type in cell.value for type in container)):
            val_replace = values[0]
        else:
            val_replace = values

        return val_replace

    def instantaite_main_obj(self, list_instantiated_obj, key, values):
        module_name = values[0][0]
        class_name = values[0][1]
        attr = values[1]
        module = importlib.import_module(module_name)
        obj_class = getattr(module, class_name)

        init_attributes = list(inspect.signature(obj_class.__init__).parameters.keys())[1:]
        list_obj = []
        for value in values[2].values():
            for k, val_attr in enumerate(value):
                if isinstance(val_attr, openpyxl.cell.cell.Cell):
                    val_replace = list_instantiated_obj[self.get_location(val_attr)[0]]
                    value[k] = self.replace_attribute(val_attr, val_replace)

            obj_data = {a: value[i] for i, a in enumerate(attr) if a in init_attributes}
            obj = obj_class(**obj_data)
            if not obj.name:
                obj.name = ""
            list_obj.append(obj)

        list_instantiated_obj[key] = list_obj
        return list_instantiated_obj

    def process_hyperlinks(self, list_instantiated_obj, key, values):
        module_name = values[0][0]
        class_name = values[0][1]
        attr = values[1]

        module = importlib.import_module(module_name)
        obj_class = getattr(module, class_name)

        init_attributes = list(inspect.signature(obj_class.__init__).parameters.keys())[1:]
        list_obj = []
        for value in values[2].values():
            for i, val in enumerate(value):
                if isinstance(val, openpyxl.cell.cell.Cell):
                    sheet_target_title, row_target  = self.get_location(val)
                    sub_module_name = self.workbook[sheet_target_title]["A2"].value
                    sub_class_name = self.workbook[sheet_target_title]["B2"].value
                    sheet_target = self.workbook[sheet_target_title]
                    sub_attr = list(sheet_target.iter_rows(min_row=3, max_row=3, min_col=2, values_only=True))[0]
                    sub_module = importlib.import_module(sub_module_name)
                    sub_obj_class = getattr(sub_module, sub_class_name)

                    moment_value = [cell.value for cell in sheet_target[int(row_target[1:]) + 2][1:]]

                    value[i] = sub_obj_class(**{a: moment_value[i] if i < len(moment_value) else None
                                                for i, a in enumerate(sub_attr)})

            obj_data = {a: value[i] for i, a in enumerate(attr) if a in init_attributes}
            obj = obj_class(**obj_data)
            if not obj.name:
                obj.name = ""
            list_obj.append(obj)

        list_instantiated_obj[key] = list_obj
        return list_instantiated_obj

    def process_other_cases(self, list_instantiated_obj, key, values, stack):
        hyperlink_list = [self.get_location(v2)[0] for val in values[2:] for v in val.values() for v2 in v if
                          isinstance(v2, openpyxl.cell.cell.Cell)]
        list_processed_list_key = list(list_instantiated_obj.keys())
        if hyperlink_list == list_processed_list_key or all(
                hyperlink in list_processed_list_key for hyperlink in hyperlink_list):
            module_name = values[0][0]
            class_name = values[0][1]
            attr = values[1]

            module = importlib.import_module(module_name)
            obj_class = getattr(module, class_name)
            list_obj = []
            for value in values[2].values():
                for k, val_attr in enumerate(value):
                    if isinstance(val_attr, openpyxl.cell.cell.Cell):
                        val_replace = list_instantiated_obj[self.get_location(val_attr)[0]]
                        value[k] = self.replace_attribute(val_attr, val_replace)

                obj_data = {a: value[i] if i < len(value) else None for i, a in enumerate(attr)}
                obj = obj_class(**obj_data)
                if not obj.name:
                    obj.name = ""
                list_obj.append(obj)

            list_instantiated_obj[key] = list_obj
        else:
            stack.extend(list({key: values}.items()))
        return list_instantiated_obj

    def process_workbook(self):
        cell_values = {}

        for sheet in self.workbook.worksheets:
            cell_value = []
            cell_value.append((sheet.cell(row=2, column=1).value, sheet.cell(row=2, column=2).value))

            attributes = []
            for value in sheet.iter_cols(min_row=3, max_row=3, min_col=2, values_only=True):
                attributes.append(value[0])

            cell_value.append(attributes)

            data = {}
            for i, value in enumerate(sheet.iter_rows(min_row=4, min_col=2, values_only=True)):
                tuple_val = []
                for j, _ in enumerate(value):
                    cell_val = sheet.cell(row=i + 4, column=j + 2)
                    if cell_val.hyperlink:
                        tuple_val.append(cell_val)
                    else:
                        tuple_val.append(cell_val.value)
                data[i] = tuple_val

            cell_value.append(data.copy())
            cell_values[sheet.title] = cell_value

        return cell_values

    def read_object(self, only_main_object: bool = True):
        list_instantiated_obj = {}
        cell_values = self.process_workbook()
        stack = list(cell_values.items())
        # if cell_values.keys().__len__() == 1:
        #     print("yes cell_values.keys().__len__() == 1")
        while stack:
            key, values = stack.pop()

            if key == self.main_sheet:
                list_instantiated_obj = self.instantaite_main_obj(list_instantiated_obj, key, values)
                break

            if any((any(isinstance(v, openpyxl.cell.cell.Cell) for v in val) for value in values[2:] for val in
                    value.values())):

                if len(values[2].keys()) > 1:
                    print("")
                    list_instantiated_obj = self.process_hyperlinks(list_instantiated_obj, key, values)
                    continue
                else:
                    list_instantiated_obj = self.process_other_cases(list_instantiated_obj, key, values, stack)

            else:
                module_name = values[0][0]
                class_name = values[0][1]
                attr = values[1]

                module = importlib.import_module(module_name)
                obj_class = getattr(module, class_name)
                list_obj = []
                for value in values[2].values():
                    obj_data = {a: value[i] if i < len(value) else None for i, a in enumerate(attr)}
                    obj = obj_class(**obj_data)
                    if not obj.name:
                        obj.name = ""
                    list_obj.append(obj)

                list_instantiated_obj[key] = list_obj

        if only_main_object:
            return list_instantiated_obj[self.main_sheet][0]
        return list_instantiated_obj

    def close(self):
        self.workbook.close()
