#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module defining recognized  file types in DC"""

import io
import openpyxl

class BinaryFile(io.BytesIO):
    extension = ''
    """Class for handling binary files with name"""

    def __init__(self, filename: str = ''):
        super().__init__()
        self.filename = filename

    def copy(self):
        """ Files deep copy  """
        file_copy = self.__class__(self.filename)
        file_copy.write(self.getbuffer())
        file_copy.seek(0)
        return file_copy

    @classmethod
    def stream_template(cls):
        template = cls()
        return template

    def save_template_to_file(self, filename):
        if self.extension and not filename.endswith(self.extension):
            filename = f'{filename}.{self.extension}'
        with open(filename, 'wb') as file:
            stream = self.stream_template()
            stream.seek(0)
            file.write(stream.getvalue())


class StringFile(io.StringIO):
    extension = ''
    """
    class for handling text files with name
    default encoding : utf-8
    """

    def __init__(self, filename: str = ''):
        super().__init__()
        self.filename = filename

    def copy(self):
        """ Files deep copy  """
        file_copy = self.__class__(self.filename)
        file_copy.write(self.getvalue())
        file_copy.seek(0)
        return file_copy

    @classmethod
    def stream_template(cls):
        template = cls()
        template.write('Text file template. Subclass the class dessia_common.files.StringFile'
                       ' to define your own filetype')
        template.seek(0)
        return template

    def save_template_to_file(self, filename):        
        if self.extension and not filename.endswith(self.extension):
            filename = f'{filename}.{self.extension}'
        with open(filename, 'w') as file:
            stream = self.stream_template()
            stream.seek(0)
            file.write(stream.getvalue())


class XLSXFile(BinaryFile):
    extension = 'xlsx'

    @classmethod
    def stream_template(cls):
        template = cls()
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title='Template of XLSX'
        ws.cell(1, 1).value = 'Subclass this class to define your own template'
        wb.save(template)
        template.seek(0)
        return template


class XLSFile(BinaryFile):
    extension = 'xls'

class XLSMFile(XLSXFile):
    extension = 'xlsm'


class TextFile(StringFile):
    extension = 'txt'


class CSVFile(StringFile):
    extension = 'csv'
    
    @classmethod
    def stream_template(cls):
        template = cls()
        template.write('"col A", "col B", "col C"\n')
        template.write('"abc", 1, 1.23\n')
        
        template.seek(0)
        return template

class MarkdownFile(StringFile):
    extension = 'md'

    @classmethod
    def stream_template(cls):
        template = cls()
        template.write('# Text file template.\n\n'
                       'Subclass the class dessia_common.files.StringFile'
                       ' to define your own filetype')
        template.seek(0)
        return template