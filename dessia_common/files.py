#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import io


class BinaryFile(io.BytesIO()):
    def __init__(self, filename: str = ''):
        io.BytesIO.__init__(self)
        self.filename = filename


class StringFile(io.StringIO()):
    def __init__(self, filename: str = ''):
        io.StringIO.__init__(self)
        self.filename = filename


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
