#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module defining recognized  file types in DC. """
import io
import openpyxl


class BinaryFile(io.BytesIO):
    """ Base class for handling binary files with name. """

    extension = ''

    def __init__(self, filename: str = ''):
        super().__init__()
        self.filename = filename

    def copy(self):
        """ Files deep copy. """
        file_copy = self.__class__(self.filename)
        file_copy.write(self.getbuffer())
        file_copy.seek(0)
        return file_copy

    @classmethod
    def stream_template(cls):
        """ Get template of class of file. """
        template = cls()
        return template

    @classmethod
    def save_template_to_file(cls, filename):
        """ Save template of class into a file. """
        if cls.extension and not filename.endswith(cls.extension):
            filename = f'{filename}.{cls.extension}'
        with open(filename, 'wb') as file:
            stream = cls.stream_template()
            stream.seek(0)
            file.write(stream.getvalue())

    def __hash__(self):
        return hash(self.filename)

    def __eq__(self, other):
        return isinstance(other, BinaryFile) and self.getbuffer() == other.getbuffer() \
               and self.filename == other.filename


class StringFile(io.StringIO):
    """ Base class for handling text files with name with default encoding : utf-8. """

    extension = ''

    def __init__(self, filename: str = ''):
        super().__init__()
        self.filename = filename

    def copy(self):
        """ Files deep copy. """
        file_copy = self.__class__(self.filename)
        file_copy.write(self.getvalue())
        file_copy.seek(0)
        return file_copy

    @classmethod
    def stream_template(cls):
        """ Get template of class of file. """
        template = cls()
        template.write('Text file template. Subclass the class dessia_common.files.StringFile'
                       ' to define your own filetype')
        template.seek(0)
        return template

    @classmethod
    def from_stream(cls, stream_file):
        """ Get a file from a stream file. """
        stream = StringFile()
        stream.write(stream_file.read().decode('utf-8'))
        stream.seek(0)
        return stream

    @classmethod
    def save_template_to_file(cls, filename):
        """ Save instantiated template into a file. """
        if cls.extension and not filename.endswith(cls.extension):
            filename = f'{filename}.{cls.extension}'
        with open(filename, 'w', encoding='utf-8') as file:
            stream = cls.stream_template()
            stream.seek(0)
            file.write(stream.getvalue())

    def __hash__(self):
        return hash(self.filename)

    def __eq__(self, other):
        return isinstance(other, StringFile) and self.getvalue() == other.getvalue() \
               and self.filename == other.filename


class XLSXFile(BinaryFile):
    """ Base class for Excel XLS files. """

    extension = 'xlsx'

    @classmethod
    def stream_template(cls):
        """ Get template of class of file. """
        template = cls()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = 'Template of XLSX'
        sheet.cell(1, 1).value = 'Subclass this class to define your own template'
        workbook.save(template)
        template.seek(0)
        return template


class XLSFile(BinaryFile):
    """ Base class for Excel files in old format. """

    extension = 'xls'


class XLSMFile(XLSXFile):
    """ Base class for Excel XLS files with macros. """

    extension = 'xlsm'


class TextFile(StringFile):
    """ Base class for text files. """

    extension = 'txt'


class CSVFile(StringFile):
    """
    Base class for Coma Separated Values (CSV) files.

    https://en.wikipedia.org/wiki/Comma-separated_values
    """

    extension = 'csv'

    @classmethod
    def stream_template(cls):
        """ Get template of class of file. """
        template = cls()
        template.write('"col A", "col B", "col C"\n')
        template.write('"abc", 1, 1.23\n')

        template.seek(0)
        return template


class MarkdownFile(StringFile):
    """
    Base class for markdown files.

    https://en.wikipedia.org/wiki/Markdown
    """

    extension = 'md'

    @classmethod
    def stream_template(cls):
        """ Get template of class of file. """
        template = cls()
        template.write('# Text file template.\n\n'
                       'Subclass the class dessia_common.files.StringFile'
                       ' to define your own filetype')
        template.seek(0)
        return template


class JsonFile(StringFile):
    """ A .json extended file. """

    extension = "json"
