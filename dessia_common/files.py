#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module defining recognized  file types in DC"""

import io
import openpyxl


class BinaryFile(io.BytesIO):
    """Class for handling binary files with name"""

    extension = ''

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

    @classmethod
    def save_template_to_file(cls, filename):
        if cls.extension and not filename.endswith(cls.extension):
            filename = f'{filename}.{cls.extension}'
        with open(filename, 'wb') as file:
            stream = cls.stream_template()
            stream.seek(0)
            file.write(stream.getvalue())


class StringFile(io.StringIO):
    """
    class for handling text files with name
    default encoding : utf-8
    """
    extension = ''

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

    @classmethod
    def save_template_to_file(cls, filename):
        if cls.extension and not filename.endswith(cls.extension):
            filename = f'{filename}.{cls.extension}'
        with open(filename, 'w', encoding='utf-8') as file:
            stream = cls.stream_template()
            stream.seek(0)
            file.write(stream.getvalue())


class XLSXFile(BinaryFile):
    """
    Excel XML
    """
    extension = 'xlsx'

    @classmethod
    def stream_template(cls):
        template = cls()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = 'Template of XLSX'
        sheet.cell(1, 1).value = 'Subclass this class to define your own template'
        workbook.save(template)
        template.seek(0)
        return template


class XLSFile(BinaryFile):
    """
    Old Excel format
    """
    extension = 'xls'


class XLSMFile(XLSXFile):
    """
    Excel XML with macros
    """
    extension = 'xlsm'


class TextFile(StringFile):
    """
    basic text file
    """
    extension = 'txt'


class CSVFile(StringFile):
    """
    Coma separated values files
    https://en.wikipedia.org/wiki/Comma-separated_values
    """
    extension = 'csv'

    @classmethod
    def stream_template(cls):
        template = cls()
        template.write('"col A", "col B", "col C"\n')
        template.write('"abc", 1, 1.23\n')

        template.seek(0)
        return template


class MarkdownFile(StringFile):
    """
    Markdown file
    https://en.wikipedia.org/wiki/Markdown
    """
    extension = 'md'

    @classmethod
    def stream_template(cls):
        template = cls()
        template.write('# Text file template.\n\n'
                       'Subclass the class dessia_common.files.StringFile'
                       ' to define your own filetype')
        template.seek(0)
        return template


class JsonFile(StringFile):
    """
    A .json extended file
    """
    extension = "json"
