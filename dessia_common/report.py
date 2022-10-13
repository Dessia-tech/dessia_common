"""
report for dessia_common

"""
from dessia_common.core import DessiaObject
from typing import List
from string import Template
import time

dessia_object_markdown_template = Template(
'''
$log
''')

class Report(DessiaObject):
    _standalone_in_db = True
    # _eq_is_data_eq = True
    # _non_serializable_attributes = []
    # _non_data_eq_attributes = ['name']
    # _non_data_hash_attributes = ['name']

    def __init__(self, name_report: str = 'Output',
                 # width_line: int = 100,
                 # time_start: float = None,
                 content: str = '',
                 # last_offset: int = 0, #To see with platform integration if taken into account
                 name: str = ''):
        """
        Name and parameters for report's layout
        """

        #Mettre en attribut facultatif pour l'export txt à chaque écriture, same pour le print

        DessiaObject.__init__(self, name=name)
        self.width_line = 100
        self.name_report = name_report
        # if time_start is None:
        self.time_start = time.time()
        # else:
        #     self.time_start = time_start
        self.last_offset = 0 #To see with platform integration if taken into account
        self.error = False
        self.content = content

    def add_time(self, offset: int = 0, text_to_add: str = '', time_to_add: float = 0):
        line = ' ' * offset
        time_elapsed = str(round((time.time() - self.time_start) / 60, 2))
        if time_to_add != 0 and text_to_add != '':
            line += f'*ELAPSED TIME min -- {text_to_add} {time_to_add} -- global {time_elapsed}*'
        else:
            line += f'*ELAPSED TIME min -- global {time_elapsed}* '
        self.add_lines([line], nb_line_blank=0)

    def open(self, option: str = 'a'):
        #To avoid : Using open without explicitly specifying an encoding
        # file = open(self.name_report + '.log', option)
        # TODO : Mettre with open
        if option == 'r':
            file = open(self.name_report + '.log', 'r')
        elif option == 'w':
            file = open(self.name_report + '.log', 'w')
        return file

    def close(self, file):
        file.close()

    def add_lines(self, lines: List[str], offset: int = 0, nb_line_blank: int = 0):
        lines += ['']*nb_line_blank
        for line in lines:
            self.content += offset*' ' + line + '  \n  '

            print(line)

        # self.close(file)
        # TODO : si besoin de l'export txt, j'ajoute que les lignes nécessaire
        # with open .... f_stream, f_stream.write en allant à la derniere ligne
        self.to_txt()

    def add_title(self, title: str):
        self.last_offset = 0
        title = title.upper()
        lines = ['# ' + title]

        self.add_lines(lines, nb_line_blank=2)

    def add_subtitle(self, title: str):
        self.last_offset = 0
        title = title.capitalize()
        # if len(title) > self.width_line:
        title = title[0: self.width_line]

        lines = ['', '## ' + title]
        self.add_lines(lines, nb_line_blank=1)

    def add_subsubtitle(self, title: str):
        offset = 2
        self.last_offset = offset
        title = title.capitalize()
        # if len(title) > self.width_line - offset:
        title = title[0: self.width_line - offset]

        lines = [' ' * offset + '### ' + title]
        self.add_lines(lines, offset, nb_line_blank=0)

    def add_text(self, text: str):
        offset = self.last_offset + 2
        line_length = self.width_line - offset

        nb_bloc = text.split(' ')
        lines = []
        line = ' ' * offset
        for bloc in nb_bloc:
            if bloc == "\n":
                lines.append(line)
                line = ' ' * offset

            elif len(line) + len(bloc) <= line_length:
                line += bloc
                if len(line) + 1 <= line_length:
                    line += ' '
            else:
                if len(line) == offset:
                    line += bloc
                    lines.append(line)
                    line = ' ' * offset
                else:
                    lines.append(line)
                    line = ' ' * offset + bloc + ' '

        lines.append(line)
        self.add_lines(lines, offset)

    def add_table(self, title: List[str], elements: List[List[any]]):

        # | Left columns  | Right columns |
        # | ------------- |:-------------:|
        # | left foo      | right foo     |
        # | left bar      | right bar     |
        # | left baz      | right baz     |

        offset = self.last_offset + 2
        # line_length = self.width_line - offset

        real_line_length = self.width_line - offset - (len(title) + 1) - 2 * len(title)
        if real_line_length < 0:
            raise KeyError('too many columns in the report')

        max_length = [len(t) for t in title]

        for element in elements:
            for index, elem in enumerate(element):
                max_length[index] = max(max_length[index], len(str(elem)))

        pourcent_length = [m / sum(max_length) for m in max_length]

        real_length = []
        for pourcent in pourcent_length[:-1]:
            real_length.append(int(pourcent * real_line_length))
        real_length.append(real_line_length - sum(real_length))

        lines = ['  \n  ']
        lines.append(self.add_table_line(offset, real_length, title))
        lines.append(self.add_table_line(offset, real_length, title, is_title = True))

        for element in elements:
            lines.append(self.add_table_line(offset, real_length, element))

        lines.append('  \n  ')
        self.add_lines(lines, offset)

    def add_table_line(self, offset, columns_length, element_to_add, is_title: bool = False):
        line_temp = ' ' * offset
        for real_l, element in zip(columns_length, element_to_add):
            full_cell_length = real_l - len(str(element))
            right_cell_length = int(full_cell_length / 2)
            left_cell_length = full_cell_length - right_cell_length
            if is_title:
                line_temp += '|:' + '-' * (right_cell_length + len(str(element)) + left_cell_length) + ':'
            else:
                line_temp += '| ' + ' ' * right_cell_length + str(element) + ' ' * left_cell_length + ' '

        line_temp += '|'
        return line_temp

    def add_error(self, text: str):
        self.error = True
        self.add_text("\n **ERROR** " + text + " \n")

    def to_txt(self):
        # TODO : connaitre le fichier comme un chemin, le réouvrir et faire ça propre
        file = self.open('w')
        file.write(self.content)
        self.close(file)

    def to_markdown(self):
        return dessia_object_markdown_template.substitute(log=self.content)
