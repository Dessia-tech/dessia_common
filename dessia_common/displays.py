import webbrowser
import os
import tempfile
import inspect

from typing import Union
from networkx import DiGraph, Graph, kamada_kawai_layout
from dessia_common.templates import visjs_template
from dessia_common.typings import JsonSerializable


class DisplaySetting:
    def __init__(self, selector, type_, method, arguments=None, serialize_data: bool = False):
        """
        Describe what method to call to get a display
        """
        self.selector = selector
        self.type = type_
        self.method = method
        if not arguments:
            arguments = {}
        self.arguments = arguments
        self.serialize_data = serialize_data

    def to_dict(self):
        """
        Serialization
        """
        return {'selector': self.selector, 'type': self.type, 'method': self.method,
                'serialize_data': self.serialize_data, 'arguments': self.arguments}

    def compose(self, attribute: str, serialize_data: bool = False):
        """
        In case of a parent getting the display settings of a children this methods allow to inject the attribute name
        to method name
        """
        return DisplaySetting(selector=self.selector, type_=self.type, method=f'{attribute}.{self.method}',
                              arguments=self.arguments, serialize_data=serialize_data)


class DisplayObject:
    def __init__(self, type_: str, data: Union[JsonSerializable, str], reference_path: str = '',
                 traceback: str = '', name: str = ''):
        """
        Container for data of display
        A traceback can be set if display fails to be generated.
        """

        self.type_ = type_
        self.data = data
        self.traceback = traceback
        self.reference_path = reference_path
        self.name = name

        if data and type_ == 'markdown':
            self.data_cleaning()

    def data_cleaning(self):
        """
        Cleanup tabs in markdown
        """
        self.data = inspect.cleandoc(self.data)

    def to_dict(self):
        """
        Simple serialization
        """
        return {'type_': self.type_, 'data': self.data, 'traceback': self.traceback,
                'reference_path': self.reference_path, 'name': self.name}


def networkx_to_visjs_data(networkx_graph: Graph):
    """
    Compute visjs data to plot from a networkx graph
    """
    visjs_data = {'name': networkx_graph.name, 'nodes': [], 'edges': []}

    pos = kamada_kawai_layout(networkx_graph)

    for i, node in enumerate(networkx_graph.nodes):
        node_dict = networkx_graph.nodes[node]
        node_data = {'id': i}

        if node in pos:
            node_data['x'], node_data['y'] = pos[node]

        if 'name' not in node_dict and 'label' not in node_dict:
            if isinstance(node, str):
                node_data['label'] = node
            elif isinstance(node, int):
                node_data['label'] = str(node)
            else:
                node_data['label'] = ''
        elif 'name' in node_dict and 'label' not in node_dict:
            node_data['label'] = node_dict['name']
        else:
            node_data['label'] = node_dict['name'] + node_dict['label']

        if 'shape' not in node_dict:
            node_data['shape'] = 'circular'
        else:
            node_data['shape'] = node_dict['shape']

        if 'color' in node_dict:
            node_data['color'] = node_dict['color']

        visjs_data['nodes'].append(node_data)

    list_nodes = list(networkx_graph.nodes)
    is_digraph = isinstance(networkx_graph, DiGraph)
    # print(is_digraph)
    for edge in networkx_graph.edges:
        index1 = list_nodes.index(edge[0])
        index2 = list_nodes.index(edge[1])
        edge_data = {'from': index1,
                     'to': index2,
                     'font': {'align': 'middle'}}

        if is_digraph:
            edge_data['arrows'] = 'to'

        visjs_data['edges'].append(edge_data)

    return visjs_data


def draw_networkx_graph(networkx_graph: Graph):
    """
    Draw a networkx graph in a browser using VisJS library
    """
    visjs_data = networkx_to_visjs_data(networkx_graph)
    content = visjs_template.substitute(**visjs_data)
    with tempfile.NamedTemporaryFile(suffix=".html",
                                     delete=False) as file:
        file.write(bytes(content, 'utf8'))

    # with open('graph_visJS.html', 'wb') as file:
    #     file.write(s.encode('utf-8'))
    webbrowser.open('file://' + os.path.realpath(file.name))
    return file.name
