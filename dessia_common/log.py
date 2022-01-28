#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from dessia_common import DessiaObject
import time
import datetime
from typing import List
import math

class Log(DessiaObject):
    _standalone_in_db = True
    _eq_is_data_eq = True
    _non_serializable_attributes = []
    _non_data_eq_attributes = ['name']
    _non_data_hash_attributes = ['name']

    def __init__(self, name_log: str = 'Output',
                 width_line: int = 100,
                 time_start: float = None,
                 name: str = ''):
        """
        A log file for based on markdown
        """
        DessiaObject.__init__(self, name=name)
        self.width_line = width_line
        self.name_log = name_log
        if time_start is None:
            self.time_start = time.time()
        else:
            self.time_start = time_start
        self.last_offset = 0

    def add_time(self, offset:int = 0):
        """
        Add a timestamp
        """
        line = ' '*offset
        line += time.asctime()
        time_elapsed = datetime.timedelta(seconds=time.time() - self.time_start)
        line += ' Time elapsed {}'.format(time_elapsed)
        return [line]


    def add_lines(self, lines: List[str], offset:int = 0):
        """
        Add some lines to log content
        """
        file = self.open()
        lines.extend(self.add_time(offset))
        lines.append('')
        for line in lines:
            file.write(line + '\n')

    def add_title(self, title: str):
        """
        Add a markdown title
        """
        self.last_offset = 0
        title = title.upper()
        if len(title) > self.width_line - 4:
            title = title[0: self.width_line - 4]

        lines = ['#'*self.width_line]
        lines.append('#' + ' '*(self.width_line-2) + '#')
        nb_white_space = self.width_line - 2 - len(title)
        space1 = int(nb_white_space/2)
        space2 = nb_white_space - space1
        lines.append('#' + ' ' * space1 + title + ' ' * space2 + '#')
        lines.append('#' + ' ' * (self.width_line - 2) + '#')
        lines.append('#'*self.width_line)
        self.add_lines(lines)

    def add_subtitle(self, title: str):
        """
        Add a markdown subtitle
        """
        self.last_offset = 0
        title = title[0].upper() + title[1:]
        if len(title) > self.width_line:
            title = title[0: self.width_line]
        lines = [title]
        lines.append('='*len(title))
        self.add_lines(lines)

    def add_subsubtitle(self, title: str):
        """
        Add a markdown subsubtitle
        """

        offset = 2
        self.last_offset = offset
        title = title[0].upper() + title[1:]
        if len(title) > self.width_line - offset:
            title = title[0: self.width_line - offset]
        lines = [' '*offset + title]
        lines.append(' '*offset + '-'*len(title))
        self.add_lines(lines, offset)

    def add_text(self, text: str):
        """
        Add some text
        """
        offset = self.last_offset + 2
        line_length = self.width_line - offset
        number_line = math.ceil(len(text)/(line_length))
        lines = []
        for i in range(number_line):
            lines.append(' '*offset + text[i*line_length: (i+1)*line_length])
        self.add_lines(lines, offset)

    def add_table(self, title: List[str], elements: List[List[any]]):
        """
        Add a markdown table to the log
        """
        offset = self.last_offset + 2
        number_column = len(title)
        line_length = self.width_line - offset
        real_line_length = line_length - (number_column + 1) - 2*number_column
        if real_line_length < 0:
            raise KeyError('too many columns in the log')

        max_length = [len(t) for t in title]

        for element in elements:
            for i, e in enumerate(element):
                max_length[i] = max(max_length[i], len(str(e)))

        sum_max_length = sum(max_length)
        pourcent_length = [m/sum_max_length for m in max_length]

        real_length = []
        for p in pourcent_length[0:-1]:
            real_length.append(int(p*real_line_length))
        actual_length = sum(real_length)
        real_length.append(real_line_length-actual_length)

        lines = []
        lines.append(' ' * offset + '-'*line_length)
        line_temp = ' '*offset
        for r, t in zip(real_length, title):
            full_cell_length = r - len(str(t))
            right_cell_length = int(full_cell_length/2)
            left_cell_length = full_cell_length - right_cell_length
            line_temp += '/ ' + ' '*right_cell_length + str(t) + ' '*left_cell_length + ' '
        line_temp += '/'
        lines.append(line_temp)
        lines.append(' ' * offset + '-' * line_length)

        for element in elements:
            line_temp = ' ' * offset
            for r, e in zip(real_length, element):
                full_cell_length = r - len(str(e))
                right_cell_length = int(full_cell_length / 2)
                left_cell_length = full_cell_length - right_cell_length
                line_temp += '/ ' + ' ' * right_cell_length + str(e) + ' ' * left_cell_length + ' '
            line_temp += '/'
            lines.append(line_temp)
            lines.append(' ' * offset + '-' * line_length)

        self.add_lines(lines, offset)

