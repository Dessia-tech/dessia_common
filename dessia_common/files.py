#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module defining recognized  file types in DC"""

import io


class BinaryFile(io.BytesIO):
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


class StringFile(io.StringIO):
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


class XLSXFile(BinaryFile):
    extension = 'xlsx'


class XLSFile(BinaryFile):
    extension = 'xls'


class TextFile(StringFile):
    extension = 'txt'


class CSVFile(StringFile):
    extension = 'csv'


class MarkdownFile(StringFile):
    extension = 'md'


class JsonFile(StringFile):
    extension = "json"
