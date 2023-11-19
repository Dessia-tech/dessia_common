#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" ExcelReader for DessiaObject. """

import inspect
from ast import literal_eval

import openpyxl

from dessia_common.utils.helpers import get_python_class_from_class_name


class ExcelReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.workbook = openpyxl.load_workbook(filepath)
        self.sheet_titles = [sheet.title for sheet in self.workbook.worksheets]
        self.main_sheet = self.workbook.worksheets[0].title

    def get_location(self, cell):
        """
        Retrieve the location of a cell's hyperlink.

        Args:
        - cell: The cell containing the hyperlink.

        Returns:
        - tuple: A tuple containing the sheet name and cell address referred to by the hyperlink.
        """
        if cell.hyperlink.location:
            target = cell.hyperlink.location.split('!')
        elif cell.hyperlink.target:
            target = cell.hyperlink.target.split('!')

        sheet_name = target[0].replace("#", '')
        if sheet_name not in self.sheet_titles:
            raise ValueError(f"The sheet '{sheet_name}' referenced by the hyperlink does not exist.")
        return sheet_name, target[1]

    def update_attribute_values(self, cell, values):
        """
        Updates attribute values based on the provided cell and values.

        Args:
        - cell: The cell containing information about the data type.
        - values: The values to be replaced.

        Returns:
        - Depending on the data type of the cell:
          - For 'Dict' type: Returns a dictionary with keys extracted from the cell
            and replaced values from the 'values' parameter.
          - For other types (excluding 'List', 'Set', 'Tuple'): Replaces the values with
            the first element of 'values' if 'values' is a list and 'cell.value' does not
            contain any of the container types.
          - Otherwise, returns 'values' itself.
        """
        container = ["Dict", "List", "Set", "Tuple"]
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

    @staticmethod
    def get_attributes_and_types(obj_class):
        """
        Retrieve attributes and their corresponding types from the __init__ method of a class.

        Args:
        - obj_class: The class object.

        Returns:
        - dict: A dictionary containing attributes and their respective types defined in the class constructor.
        """
        init_signature = inspect.signature(obj_class.__init__)
        init_parameters = init_signature.parameters

        attribute_types = {}

        for attr_name, param in init_parameters.items():
            if attr_name != 'self':
                attribute_types[attr_name] = param.annotation if param.annotation != param.empty else None

        return attribute_types

    @staticmethod
    def get_data(value, attributes, init_attributes):
        """
        Extracts relevant data based on attributes and initial attributes for object instantiation.

        Args:
        - value (list): List containing values corresponding to attributes.
        - attributes (list): List of attributes related to the object.
        - init_attributes (list): List of initial attributes for object instantiation.

        Returns:
        - dict: Dictionary containing extracted object data based on matching attributes and initial attributes.
        """
        object_data = {}
        for i, attribute in enumerate(attributes):
            if attribute in init_attributes:
                if isinstance(value[i], str):
                    try:
                        object_data[attribute] = literal_eval(value[i])
                    except (ValueError, SyntaxError):
                        object_data[attribute] = value[i]
                else:
                    object_data[attribute] = value[i]
        return object_data

    def instantiate_main_objects(self, instantiated_objects, key, values):
        """
        Instantiate main objects based on provided values and attributes.

        Args:
        - instantiated_objects_list (dict): Dictionary containing instantiated objects.
        - key (str): Key to identify the instantiated objects.
        - values (list): List containing module name, class name, and attributes.

        Returns:
        - dict: Updated dictionary of instantiated objects.
        """

        module_name = values[0][0]
        class_name = values[0][1]
        object_attributes = values[1]
        full_class_name = f"{module_name}.{class_name}"
        obj_class = get_python_class_from_class_name(full_class_name=full_class_name)

        initial_attributes = self.get_attributes_and_types(obj_class)
        objects = []

        for value_set in values[2].values():
            for index, attr_value in enumerate(value_set):
                if isinstance(attr_value, openpyxl.cell.cell.Cell):
                    replaced_value = instantiated_objects[self.get_location(attr_value)[0]]
                    value_set[index] = self.update_attribute_values(attr_value, replaced_value)

            object_data = self.get_data(value_set, object_attributes, initial_attributes)
            object_ = obj_class(**object_data)
            if not object_.name:
                object_.name = ""
            objects.append(object_)

        instantiated_objects[key] = objects
        return instantiated_objects

    def process_multiple_hyperlink_rows(self, instantiated_objects, key, values):
        """
        Process multiple rows in a sheet, each representing an object with hyperlinks.

        Args:
        - instantiated_objects (dict): Dictionary containing instantiated objects.
        - key (str): Key to associate with the list of instantiated objects.
        - values (list): List containing information to process.

        Returns:
        - dict: Updated dictionary of instantiated objects.
        """
        module_name = values[0][0]
        class_name = values[0][1]
        object_attributes = values[1]

        full_class_name = f"{module_name}.{class_name}"
        obj_class = get_python_class_from_class_name(full_class_name=full_class_name)

        initial_attributes = self.get_attributes_and_types(obj_class)
        objects = []
        for value_set in values[2].values():
            for i, val in enumerate(value_set):
                if isinstance(val, openpyxl.cell.cell.Cell):
                    sheet_target_title, row_target = self.get_location(val)
                    sub_module_name = self.workbook[sheet_target_title]["A2"].value
                    sub_class_name = self.workbook[sheet_target_title]["B2"].value
                    sheet_target = self.workbook[sheet_target_title]
                    sub_attr = list(sheet_target.iter_rows(min_row=3, max_row=3, min_col=2, values_only=True))[0]
                    sub_full_class_name = f"{sub_module_name}.{sub_class_name}"
                    sub_obj_class = get_python_class_from_class_name(full_class_name=sub_full_class_name)
                    sub_init_attributes = self.get_attributes_and_types(sub_obj_class)

                    moment_value = [cell.value for cell in sheet_target[int(row_target[1:]) + 2][1:]]
                    value_set[i] = sub_obj_class(**(self.get_data(moment_value, sub_attr, sub_init_attributes)))

            object_data = self.get_data(value_set, object_attributes, initial_attributes)
            object_ = obj_class(**object_data)
            if not object_.name:
                object_.name = ""
            objects.append(object_)

        instantiated_objects[key] = objects
        return instantiated_objects

    def process_single_hyperlink_row(self, instantiated_objects, key, values, stack):
        """
        Process a single row in a sheet representing an object with hyperlinks.

        Args:
        - instantiated_objects (dict): Dictionary containing instantiated objects.
        - key (str): Key to associate with the list of instantiated objects.
        - values (list): List containing information to process.
        - stack (list): Stack to handle unprocessed items.

        Returns:
        - dict: Updated dictionary of instantiated objects.
        """
        hyperlink_list = [self.get_location(v2)[0] for val in values[2:] for v in val.values() for v2 in v if
                          isinstance(v2, openpyxl.cell.cell.Cell)]
        list_processed_list_key = list(instantiated_objects.keys())
        if hyperlink_list == list_processed_list_key or all(
                hyperlink in list_processed_list_key for hyperlink in hyperlink_list):
            module_name = values[0][0]
            class_name = values[0][1]
            object_attributes = values[1]

            full_class_name = f"{module_name}.{class_name}"
            obj_class = get_python_class_from_class_name(full_class_name=full_class_name)
            initial_attributes = self.get_attributes_and_types(obj_class)

            list_obj = []
            for value_set in values[2].values():
                for k, val_attr in enumerate(value_set):
                    if isinstance(val_attr, openpyxl.cell.cell.Cell):
                        val_replace = instantiated_objects[self.get_location(val_attr)[0]]
                        value_set[k] = self.update_attribute_values(val_attr, val_replace)

                object_data = self.get_data(value_set, object_attributes, initial_attributes)
                obj = obj_class(**object_data)
                if not obj.name:
                    obj.name = ""
                list_obj.append(obj)

            instantiated_objects[key] = list_obj
        else:
            stack.extend(list({key: values}.items()))
        return instantiated_objects

    def process_simple_sheet(self, instantiated_objects, key, values):
        """
        Process a sheet without hyperlinks containing simple cell values to instantiate objects.

        Args:
        instantiated_objects (dict): A dictionary containing instantiated objects.
        key (str): The key representing the current sheet being processed.
        values (list): A list of values containing module name, class name, attributes, and data.

        Returns:
        dict: The updated dictionary of instantiated objects after processing the sheet.
        """
        module_name = values[0][0]
        class_name = values[0][1]
        object_attributes = values[1]
        full_class_name = f"{module_name}.{class_name}"
        obj_class = get_python_class_from_class_name(full_class_name=full_class_name)

        initial_attributes = self.get_attributes_and_types(obj_class)
        objects = []
        for value_set in values[2].values():
            object_data = self.get_data(value_set, object_attributes, initial_attributes)
            object_ = obj_class(**object_data)
            if not object_.name:
                object_.name = ""
            objects.append(object_)

        instantiated_objects[key] = objects
        return instantiated_objects

    def process_workbook(self):
        """
        Process the workbook's sheets and extract information.

        This function iterates through each sheet in the workbook and gathers specific data from each sheet.
        It collects module and class names along with attributes and associated data.

        Returns:
        dict: A dictionary containing information extracted from the workbook sheets.
              Each key represents the sheet title, and its value is a list consisting of:
              - Tuple containing module and class names extracted from cells (row=2, column=1) and (row=2, column=2).
              - List of attributes extracted from row 3, starting from column 2.
              - Dictionary containing row-wise data (starting from row 4, column 2) with column values as per
               attributes.
        """
        extracted_data = {}

        for sheet in self.workbook.worksheets:
            sheet_data = []
            sheet_data.append((sheet.cell(row=2, column=1).value, sheet.cell(row=2, column=2).value))

            attributes = []
            for value in sheet.iter_cols(min_row=3, max_row=3, min_col=2, values_only=True):
                attributes.append(value[0])

            sheet_data.append(attributes)

            row_data = {}
            for i, value in enumerate(sheet.iter_rows(min_row=4, min_col=2, values_only=True)):
                column_values = []
                for j, _ in enumerate(value):
                    cell_value = sheet.cell(row=i + 4, column=j + 2)
                    if cell_value.hyperlink:
                        column_values.append(cell_value)
                    else:
                        column_values.append(cell_value.value)
                row_data[i] = column_values

            sheet_data.append(row_data.copy())
            extracted_data[sheet.title] = sheet_data

        return extracted_data

    def read_object(self):
        """
        Read and process data to instantiate objects from the workbook.

        Reads the processed workbook data, instantiates objects, and organizes them based on the provided classes and
        modules.

        Note:
        This method does not support processing variables of complex types, such as `List[Tuple[float, float]]`, due to
        limitations in interpreting complex nested structures within Excel sheets.

        Returns:
        object: The instantiated object obtained from the main sheet's data.
        """
        instantiated_objects = {}
        cell_values = self.process_workbook()
        stack = list(cell_values.items())

        while stack:
            key, values = stack.pop()

            if key == self.main_sheet:
                instantiated_objects = self.instantiate_main_objects(instantiated_objects, key, values)
                break

            is_cell_instance = any(
                (any(isinstance(cell_value, openpyxl.cell.cell.Cell) for cell_value in val) for value in values[2:] for
                 val in value.values()))
            if is_cell_instance:

                nb_objects_in_sheet = len(values[2].keys())
                if nb_objects_in_sheet > 1:
                    instantiated_objects = self.process_multiple_hyperlink_rows(instantiated_objects, key, values)
                else:
                    instantiated_objects = self.process_single_hyperlink_row(instantiated_objects, key, values, stack)

            else:
                instantiated_objects = self.process_simple_sheet(instantiated_objects, key, values)

        return instantiated_objects[self.main_sheet][0]

    def close(self):
        """
        Closes the workbook.

        This method closes the associated workbook, ensuring any changes made are saved and resources are released.
        """
        self.workbook.close()
