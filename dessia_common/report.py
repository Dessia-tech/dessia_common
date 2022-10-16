"""
report for dessia_common

"""
from dessia_common.core import DessiaObject
from typing import List
from string import Template
import time
import inspect
import networkx as nx
import numpy as npy
import math

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
        self.profiler = Profiler()

    def add_time(self, offset: int = 0, text_to_add: str = '', time_to_add: float = 0):
        line = ' ' * offset
        time_elapsed = str(round((time.time() - self.time_start) / 60, 2))
        if time_to_add != 0 and text_to_add != '':
            line += f'*ELAPSED TIME min -- {text_to_add} {time_to_add} -- global {time_elapsed}*'
        else:
            line += f'*ELAPSED TIME min -- global {time_elapsed}* '
        self.add_lines([line], nb_line_blank=0)

    # def open(self, option: str = 'a'):
    #     #To avoid : Using open without explicitly specifying an encoding
    #     # file = open(self.name_report + '.log', option)
    #     # TODO : Mettre with open
    #     if option == 'r':
    #         file = open(self.name_report + '.log', 'r')
    #     elif option == 'w':
    #         file = open(self.name_report + '.log', 'w')
    #     return file
    #
    # def close(self, file):
    #     file.close()

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

    def open(self, option: str = 'a'):
        file = open(self.name_report + '.log', option)
        return file

    def close(self, file):
        file.close()

    def to_txt(self):
        # TODO : connaitre le fichier comme un chemin, le réouvrir et faire ça propre
        file = self.open('w')
        file.write(self.content)
        self.close(file)

    def to_markdown(self):
        return dessia_object_markdown_template.substitute(log=self.content)

    def open_event(self, _stack, _self, specific_name:str = ''):
        self.new_event(_stack, _self, 'open', specific_name = specific_name)

    def close_event(self, _stack, _self, specific_name:str = ''):
        self.new_event(_stack, _self, 'closed', specific_name=specific_name)
        self.export_profiler(cut_time=5)
        self.export_final_profiler()

    def new_event(self, _stack, _self, status, specific_name: str = ''):
        name_method = str(_stack[0][3])
        name_temp = str(_self.__class__).split('\'')[1]
        if name_temp == 'type':
            name_temp = str(_self).split('\'')[1]
        if name_method == '__init__':
            name = name_temp
        else:
            name = name_temp + '.' + name_method
        t = time.time()
        if hasattr(_self, 'name'):
            if _self.name is not '':
                specific_name = _self.name
        self.profiler.add_event(name, t, status, specific_name = specific_name)

    def export_profiler(self, cut_time: float = 5):
        file = open(self.name_report +'_1' + '.log', 'w')
        graph = self.profiler.genere_summary_graph(self.profiler.graph, cut_time=cut_time)
        self.profiler.update_graph_with_non_follow(graph)
        content = self.profiler.genere_log(graph)
        file.write(content)
        file.close()

    def export_final_profiler(self):
        file = open(self.name_report +'_2' + '.log', 'w')
        graph = self.profiler.graph.copy()
        self.profiler.update_graph_with_non_follow(graph)
        content = self.profiler.genere_final_log(graph)
        file.write(content)
        file.close()

class Event(DessiaObject):
    _standalone_in_db = False
    _eq_is_data_eq = True
    _non_data_eq_attributes = ['duration', 'name']

    def __init__(self, name_event: str, time: int,
                 duration: int = None, name: str = '',
                 level: int = 0):
        self.name_event = name_event
        self.time = time
        self.level = level
        if duration:
            self.duration = duration
        else:
            self.duration = math.inf
        DessiaObject.__init__(self, name=name)

    def update_duration(self, time):
        self.duration = time - self.time

    def __lt__(self, other):
        # define for the sorted function
        if self.time < other.time:
            return True
        elif self.time == other.time:
            if self.level < other.level:
                return True
            else:
                return False
        else:
            return False

class Profiler(DessiaObject):
    _standalone_in_db = True
    _non_serializable_attributes = ['graph']

    def __init__(self, events: List[Event] = None, name: str = ''):
        if events:
            self.events = events
        else:
            self.events = []
        self.graph = nx.DiGraph()
        event_init = Event('ground', time.time(), level = 1)
        self.event_init = event_init
        self.dict_open_event = {'ground': event_init}
        self.last_open_event = event_init
        self.current_node = event_init
        DessiaObject.__init__(self, name=name)

    def add_event(self, name: str, time: int, status: str,
                  specific_name: str = ''):
        if status == 'closed':
            self.current_node = self.dict_open_event[name]
            self.current_node.update_duration(time)
            self.last_open_event = list(self.graph.predecessors(self.current_node))[0]
            del self.dict_open_event[name]

        elif status == 'open':
            if self.event_init == self.last_open_event:
                level = 2
            else:
                path = nx.shortest_path(self.graph, self.event_init, self.last_open_event)
                level = len(path) + 1
            self.current_node = Event(name, time, name=specific_name, level = level)
            self.graph.add_edge(self.last_open_event, self.current_node)
            self.last_open_event = self.current_node
            self.dict_open_event[name] = self.current_node

    def write_event_path(self, graph, current_node:Event):
        path = nx.shortest_path(graph, self.event_init, current_node)
        short_name = path[-1].name_event.split('.')[-1]
        duration = str(round(path[-1].duration, 4))
        if path[-1].name is not '':
            string = duration + ' - ' + short_name + ' [{}]'.format(path[-1].name)
        else:
            string = duration + ' - ' + short_name
        return string

    def genere_log(self, graph):
        string = ''
        nodes = list(graph.nodes)
        nodes.sort()
        for node in nodes:
            shift_space = node.level - 1
            string += ' '*shift_space + self.write_event_path(graph, node) +'\n'
        return string

    def update_graph_with_non_follow(self, graph):
        nodes = list(graph.nodes)
        for node in nodes:
            if node.duration:
                delta1_init = node.time
                end_time = delta1_init + node.duration
                next_nodes = list(graph[node].keys())
                next_nodes.sort()

                check = True
                for n_node in next_nodes:
                    if not n_node.duration:
                        check = False
                if len(next_nodes) == 0:
                    check = False

                if check:
                    for n_node in next_nodes:
                        delta2_init = n_node.time
                        delta1 = delta2_init - delta1_init
                        delta2 = n_node.duration
                        if delta1 > 1e-10:
                            if self.event_init == node:
                                level = 2
                            else:
                                path = nx.shortest_path(graph, self.event_init, node)
                                level = len(path) + 1
                            new_node = Event('_other', delta1_init, delta1, level = level)
                            graph.add_edge(node, new_node)
                        delta1_init = delta2_init + delta2
                    if end_time - delta1_init > 1e-10:
                        if self.event_init == node:
                            level = 2
                        else:
                            path = nx.shortest_path(graph, self.event_init, node)
                            level = len(path) + 1
                        new_node = Event('_other', delta1_init, end_time - delta1_init, level = level)
                        graph.add_edge(node, new_node)

    def genere_summary_graph(self, graph, cut_time = 5):
        _show_graph = nx.DiGraph()
        dfs_tree = nx.dfs_tree(graph, self.event_init)
        for (event1, event2) in dfs_tree.edges():
            if event1.duration:
                if event1.duration > cut_time:
                    _show_graph.add_edge(event1, event2)
        return _show_graph

    def genere_final_log(self, graph):
        string = ''
        nodes = list(graph.nodes)
        dict_log = {}
        for node in nodes:
            if node.duration > 1e20:
                continue
            if not graph[node]:
                if node.name_event in dict_log:
                    dict_log[node.name_event] += node.duration
                else:
                    dict_log[node.name_event] = node.duration

        dict_name_index = {n: i for i, (n, d) in enumerate(dict_log.items())}
        dict_index_name = {i: n for n, i in dict_name_index.items()}
        max_index = max(list(dict_index_name.keys()))
        max_len_name = max([len(k) for k in list(dict_log.keys())])
        list_duration_with_index = [dict_log[dict_index_name[i]] for i in range(max_index + 1)]
        indice_sort = npy.argsort(npy.array(list_duration_with_index))
        sum_duration = sum([d for d in list_duration_with_index if d < 1e20])

        for i in indice_sort[::-1]:
            duration = dict_log[dict_index_name[i]]
            pourcentage = round(100*(duration/sum_duration), 2)
            len_pourcentage = 10
            string += dict_index_name[i] + ' '*(max_len_name - len(dict_index_name[i])) + ' - ' + str(pourcentage) + '%' + ' '*(len_pourcentage - len(str(pourcentage))) + ' - ' + str(duration) + '\n'
        return string

