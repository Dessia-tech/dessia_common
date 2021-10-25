import webbrowser
import os
from dessia_common.templates import visjs_template
from networkx import DiGraph


def networkx_to_visjs_data(networkx_graph):
    visjs_data = {'name':networkx_graph.name, 'nodes': [], 'edges': []}

    for i, node in enumerate(networkx_graph.nodes):
        node_dict = networkx_graph.nodes[node]
        node_data = {'id': i}

        if 'name' not in node_dict and 'label' not in node_dict:
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
    for edge in networkx_graph.edges:
        index1 = list_nodes.index(edge[0])
        index2 = list_nodes.index(edge[1])
        edge_data = {'from': index1,
                     'to': index2,
                     'font': {'align': 'middle'}}

        if is_digraph:
            edge_data['arrow'] = 'to'

        visjs_data['edges'].append(edge_data)

    return visjs_data


def draw_networkx_graph(networkx_graph):
    visjs_data = networkx_to_visjs_data(networkx_graph)
    s = visjs_template.substitute(**visjs_data)
    with open('graph_visJS.html', 'wb') as file:
        file.write(s.encode('utf-8'))
    webbrowser.open('file://' + os.path.realpath('graph_visJS.html'))
