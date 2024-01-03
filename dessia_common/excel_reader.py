#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" ExcelReader for DessiaObject. """

import inspect
from ast import literal_eval
from typing import Dict, Tuple, List

import openpyxl

from dessia_common.utils.helpers import get_python_class_from_class_name


def missed_attribute(attributes, init_attributes):
    """
    Check for missing attributes in a dictionary of current attributes compared to initial attributes.

    Args:
    - attributes (list or set): The current attributes to be checked.
    - init_attributes (dict): A dictionary containing initial attributes as keys and their details as values.

    Returns:
    - tuple: A tuple containing two elements:
        - A boolean indicating whether all attributes are present or not.
        - If an attribute is missing with a `None` default value, it returns False along with the missing attribute
         name.
    """
    if len(attributes) != len(init_attributes.keys()):
        if len(attributes) > len(init_attributes.keys()):
            missing_attributes = set(init_attributes.keys()) - set(attributes)
        else:
            missing_attributes = set(init_attributes.keys()) - set(attributes)
        for attr_name in missing_attributes:
            if init_attributes[attr_name]['default_value'] == "empty":
                return False, attr_name
    return True, None


def check_duplicate_attributes(attributes):
    """
    Check for the presence of duplicate attributes in a list of attribute names.

    Args:
    - attributes (list): A list containing attribute names to be checked.

    Returns:
    - str or None: If duplicates are found, returns the first duplicated attribute name; otherwise, returns None.
    """
    seen = set()
    for attribute in attributes:
        if attribute in seen:
            return attribute
        seen.add(attribute)
    return None


class ExcelDataExtract:
    """Data extracted from sheet of Excel file."""

    def __init__(self, class_info: Tuple[str, str], attributes: List[str], datas):
        self.class_info = class_info
        self.attributes = attributes
        self.datas = datas


class ExcelDatasExtracted:
    """Collection of data extracted from Excel."""

    def __init__(self, extracted_datas: Dict[str, ExcelDataExtract]):
        self.extracted_datas = extracted_datas


class ExcelReader:
    def __init__(self, stream):
        self.stream = stream
        self.workbook = openpyxl.load_workbook(stream, data_only=True)
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

        else:
            raise ValueError("The provided cell does not contain a valid hyperlink location or target.")

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
            actual_value = {}
            sheet = self.workbook[target]
            column = sheet["A"]
            for row in column[3:]:
                dict_keys.append(row.value.split('.')[1])

            for key, value in zip(dict_keys, values):
                actual_value[key] = value

            return actual_value

        if isinstance(values, list) and (not any(type_ in cell.value for type_ in container)):
            actual_value = values[0]
        else:
            actual_value = values

        return actual_value

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

        attribute_info = {}

        for attr_name, param in init_parameters.items():
            if attr_name != 'self':
                attr_info = {
                    'type': param.annotation if param.annotation != param.empty else None,
                    'default_value': param.default if param.default != param.empty else "empty"
                }
                attribute_info[attr_name] = attr_info

        return attribute_info

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
        detect_missing_attribute, attribute = missed_attribute(attributes, init_attributes)
        if not detect_missing_attribute:
            raise ValueError(f"Missing attribute '{attribute}' doesn't have a default value.")

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

    def create_object(self, obj_class, value_set, attributes, initial_attributes):
        """Create an object of a specified class using provided data."""
        object_data = self.get_data(value_set, attributes, initial_attributes)
        object_ = obj_class(**object_data)
        if not object_.name:
            object_.name = ""

        return object_

    def instantiate_main_object(self, instantiated_objects, class_info, attributes, key, values):
        """
        Instantiate main objects based on provided values and attributes.

        Args:
        - instantiated_objects (dict): Dictionary containing instantiated objects.
        - key (str): Key to identify the instantiated objects.
        - values (list): List containing module name, class name, and attributes.

        Returns:
        - dict: Updated dictionary of instantiated objects.
        """

        object_class = f"{class_info[0]}.{class_info[1]}"
        class_ = get_python_class_from_class_name(full_class_name=object_class)

        initial_attributes = self.get_attributes_and_types(class_)
        objects = []

        value_set = list(values.values())[0]
        for index, value in enumerate(value_set):
            if isinstance(value, openpyxl.cell.cell.Cell):
                replaced_value = instantiated_objects[self.get_location(value)[0]]
                value_set[index] = self.update_attribute_values(value, replaced_value)

        objects.append(self.create_object(class_, value_set, attributes, initial_attributes))

        instantiated_objects[key] = objects
        return instantiated_objects

    def process_multiple_hyperlink_rows(self, instantiated_objects, class_info, attributes, key, values):
        """
        Process multiple rows in a sheet, each representing an object with hyperlinks.

        Args:
        - instantiated_objects (dict): Dictionary containing instantiated objects.
        - key (str): Key to associate with the list of instantiated objects.
        - values (list): List containing information to process.

        Returns:
        - dict: Updated dictionary of instantiated objects.
        """

        object_class = f"{class_info[0]}.{class_info[1]}"
        class_ = get_python_class_from_class_name(full_class_name=object_class)

        initial_attributes = self.get_attributes_and_types(class_)
        objects = []
        for value_set in values.values():
            for i, cell in enumerate(value_set):
                if isinstance(cell, openpyxl.cell.cell.Cell):
                    sheet_target_title, row_target = self.get_location(cell)
                    sub_module_name = self.workbook[sheet_target_title]["A2"].value
                    sub_class_name = self.workbook[sheet_target_title]["B2"].value
                    sheet_target = self.workbook[sheet_target_title]
                    sub_attr = list(sheet_target.iter_rows(min_row=3, max_row=3, min_col=2, values_only=True))[0]
                    sub_full_class_name = f"{sub_module_name}.{sub_class_name}"
                    sub_obj_class = get_python_class_from_class_name(full_class_name=sub_full_class_name)
                    sub_init_attributes = self.get_attributes_and_types(sub_obj_class)

                    moment_value = [cell.value for cell in sheet_target[int(row_target[1:]) + 2][1:]]
                    value_set[i] = sub_obj_class(**(self.get_data(moment_value, sub_attr, sub_init_attributes)))

            objects.append(self.create_object(class_, value_set, attributes, initial_attributes))

        instantiated_objects[key] = objects
        return instantiated_objects

    def process_single_hyperlink_row(self, instantiated_objects, class_info, attributes, key, values):
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

        object_class = f"{class_info[0]}.{class_info[1]}"

        class_ = get_python_class_from_class_name(full_class_name=object_class)
        initial_attributes = self.get_attributes_and_types(class_)

        objects = []
        value_set = list(values.values())[0]
        for k, cell in enumerate(value_set):
            if isinstance(cell, openpyxl.cell.cell.Cell):
                val_replace = instantiated_objects[self.get_location(cell)[0]]
                value_set[k] = self.update_attribute_values(cell, val_replace)

        objects.append(self.create_object(class_, value_set, attributes, initial_attributes))

        instantiated_objects[key] = objects

        return instantiated_objects

    def process_simple_sheet(self, instantiated_objects, class_info, attributes, key, values):
        """
        Process a sheet without hyperlinks containing simple cell values to instantiate objects.

        Args:
        instantiated_objects (dict): A dictionary containing instantiated objects.
        key (str): The key representing the current sheet being processed.
        values (list): A list of values containing module name, class name, attributes, and data.

        Returns:
        dict: The updated dictionary of instantiated objects after processing the sheet.
        """

        object_class = f"{class_info[0]}.{class_info[1]}"
        class_ = get_python_class_from_class_name(full_class_name=object_class)

        initial_attributes = self.get_attributes_and_types(class_)
        objects = []
        for value_set in values.values():
            objects.append(self.create_object(class_, value_set, attributes, initial_attributes))

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
        extracted_datas = {}

        for sheet in self.workbook.worksheets:
            class_info = (sheet.cell(row=2, column=1).value, sheet.cell(row=2, column=2).value)

            attributes = []
            for value in sheet.iter_cols(min_row=3, max_row=3, min_col=2, values_only=True):
                attributes.append(value[0])

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

            extracted_datas[sheet.title] = ExcelDataExtract(class_info=class_info, attributes=attributes,
                                                            datas=row_data.copy())

        return ExcelDatasExtracted(extracted_datas=extracted_datas)

    def read_workbook(self):
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
        data_structure = self.process_workbook()
        stack = list(data_structure.extracted_datas.items())

        while stack:
            key, extracted_data = stack.pop()

            if key == self.main_sheet:
                instantiated_objects = self.instantiate_main_object(instantiated_objects, extracted_data.class_info,
                                                                    extracted_data.attributes, key,
                                                                    extracted_data.datas)
                break

            is_cell_instance = any((any(isinstance(cell_value, openpyxl.cell.cell.Cell) for cell_value in val) for val
                                    in extracted_data.datas.values()))
            if is_cell_instance:

                nb_objects_in_sheet = len(extracted_data.datas.keys())
                if nb_objects_in_sheet > 1:
                    instantiated_objects = self.process_multiple_hyperlink_rows(instantiated_objects,
                                                                                extracted_data.class_info,
                                                                                extracted_data.attributes,
                                                                                key, extracted_data.datas)

                else:
                    instantiated_objects = self.process_single_hyperlink_row(instantiated_objects,
                                                                             extracted_data.class_info,
                                                                             extracted_data.attributes,
                                                                             key, extracted_data.datas)

            else:
                instantiated_objects = self.process_simple_sheet(instantiated_objects, extracted_data.class_info,
                                                                 extracted_data.attributes, key, extracted_data.datas)


        return instantiated_objects[self.main_sheet][0]

    def process_workbook_catalog(self):
        """
        Process the workbook's sheets and extract information (Catalog).

        This function iterates through each sheet in the workbook and gathers specific data from each sheet.
        It collects module and class names along with attributes and associated data.

        Returns:
        dict: A dictionary containing information extracted from the workbook sheets.
              Each key represents the sheet title, and its value is a list consisting of:
              - Tuple containing module and class names extracted from cells (row=1, column=1) and (row=1, column=2).
              - List of attributes extracted from row 2, starting from column 1.
              - Dictionary containing row-wise data (starting from row 3, column 1) with column values as per
               attributes.
        """

        extracted_datas = {}
        for sheet in self.workbook.worksheets:
            class_info = (sheet.cell(row=1, column=1).value, sheet.cell(row=1, column=2).value)

            attributes = []
            for value in sheet.iter_cols(min_row=2, max_row=2, min_col=1, values_only=True):
                attributes.append(value[0])

            duplicate_attribute = check_duplicate_attributes(attributes)
            if duplicate_attribute:
                raise ValueError(f"Duplicate attribute '{duplicate_attribute}' detected. Each attribute name should "
                                 f"be unique.")

            row_data = {}
            for i, value in enumerate(sheet.iter_rows(min_row=3, min_col=1, values_only=True)):
                column_values = []
                for j, _ in enumerate(value):
                    cell_value = sheet.cell(row=i + 3, column=j + 1)
                    column_values.append(cell_value.value)
                row_data[i] = column_values

            extracted_datas[sheet.title] = ExcelDataExtract(class_info=class_info, attributes=attributes,
                                                            datas=row_data.copy())

        return ExcelDatasExtracted(extracted_datas=extracted_datas)

    def read_catalog(self):
        """
        Read and process data to instantiate objects from the workbook (Catalog).

        This method retrieves cell values from a workbook, processes them, and creates instantiated
        objects based on the processed data.

        Returns:
        instantiated_objects (dict): A dictionary containing instantiated objects created from
        the processed workbook data.
        """
        instantiated_objects = {}
        data_structure = self.process_workbook_catalog()
        stack = list(data_structure.extracted_datas.items())

        while stack:
            key, extracted_data = stack.pop()
            instantiated_objects = self.process_simple_sheet(instantiated_objects, extracted_data.class_info,
                                                             extracted_data.attributes, key, extracted_data.datas)

        return instantiated_objects

    def close(self):
        """
        Closes the workbook.

        This method closes the associated workbook, ensuring any changes made are saved and resources are released.
        """
        self.workbook.close()
