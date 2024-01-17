""" Displays for dessia_common. """
import webbrowser
import os
import tempfile
import inspect
import json


from typing import Union, Optional
from networkx import DiGraph, Graph, kamada_kawai_layout
from dessia_common.templates import visjs_template
from dessia_common.typings import JsonSerializable


class DisplaySetting:
    """ Describe which method to call to get a display. """

    def __init__(self, selector: Optional[str], type_: str, method: str, arguments=None,
                 serialize_data: bool = False, load_by_default: bool = False):
        self.selector = selector
        self.type = type_
        self.method = method
        if not arguments:
            arguments = {}
        self.arguments = arguments
        self.serialize_data = serialize_data
        self.load_by_default = load_by_default

    @property
    def reference_path(self) -> str:
        """
        Get reference path or set it to its default value if not defined.

        Experimental.
        """
        if self.arguments is None:
            return "#"

        reference_path = self.arguments.get("reference_path", "#")
        if reference_path is None:
            return "#"
        return reference_path

    def to_dict(self):
        """ Serialization: make a dict from class instance attributes. """
        return {"selector": self.selector, "type": self.type, "method": self.method, "arguments": self.arguments,
                "serialize_data": self.serialize_data, "load_by_default": self.load_by_default}

    def compose(self, attribute: str):
        """
        Handles deep calls to method.

        In case of a parent getting the display settings of a children, this method allow
        to inject the attribute name to method name.
        """
        self.method = f"{attribute}.{self.method}"


class DisplayObject:
    """ Container for data of display. A traceback can be set if display fails to be generated. """

    def __init__(self, type_: str, data: Union[JsonSerializable, str],
                 reference_path: str = '', traceback: str = '', name: str = ''):
        self.type_ = type_
        self.data = data
        self.traceback = traceback
        self.reference_path = reference_path
        self.name = name

        if data and type_ == "markdown":
            self.data_cleaning()

    def data_cleaning(self):
        """ Cleanup tabs in markdown. """
        self.data = inspect.cleandoc(self.data)

    def to_dict(self):
        """ Simple serialization. """
        return {"type_": self.type_, "data": self.data, "traceback": self.traceback,
                "reference_path": self.reference_path, "name": self.name}


def networkx_to_visjs_data(networkx_graph: Graph):
    """ Compute visjs data to plot from a networkx graph. """
    visjs_data = {"name": networkx_graph.name, "nodes": [], "edges": []}

    pos = kamada_kawai_layout(networkx_graph)

    for i, node in enumerate(networkx_graph.nodes):
        node_dict = networkx_graph.nodes[node]
        node_data = {"id": i}

        if node in pos:
            node_data["x"], node_data["y"] = pos[node]

        if "name" not in node_dict and "label" not in node_dict:
            if isinstance(node, str):
                node_data["label"] = node
            elif isinstance(node, int):
                node_data["label"] = str(node)
            else:
                node_data["label"] = ''
            node_data["title"] = node_data["label"]
            if len(node_data["label"]) > 10:
                node_data["label"] = "[ ]"
        elif "name" in node_dict and "label" not in node_dict:
            node_data["label"] = node_dict["name"]
        else:
            node_data["label"] = node_dict["name"] + node_dict["label"]

        if "shape" not in node_dict:
            node_data["shape"] = "circular"
        else:
            node_data["shape"] = node_dict["shape"]

        if "color" in node_dict:
            node_data["color"] = node_dict["color"]
        visjs_data["nodes"].append(node_data)

    list_nodes = list(networkx_graph.nodes)
    is_digraph = isinstance(networkx_graph, DiGraph)
    for edge in networkx_graph.edges:
        index1 = list_nodes.index(edge[0])
        index2 = list_nodes.index(edge[1])
        edge_nx_data = networkx_graph.get_edge_data(*edge)
        edge_data = {"from": index1, "to": index2, "font": {"align": "middle"}}
        if is_digraph:
            if "head_type" in edge_nx_data:
                edge_data["arrows"] = {"to": {"enabled": True, "type": edge_nx_data["head_type"]}}
            else:
                edge_data["arrows"] = "to"
        if "color" in edge_nx_data:
            edge_data["color"] = {"color": edge_nx_data["color"]}
        visjs_data["edges"].append(edge_data)
    return visjs_data


def draw_networkx_graph(networkx_graph: Graph):
    """ Draw a networkx graph in a browser using VisJS library. """
    visjs_data = networkx_to_visjs_data(networkx_graph)
    content = visjs_template.substitute(nodes=json.dumps(visjs_data["nodes"]),
                                        edges=json.dumps(visjs_data["edges"]),
                                        name=visjs_data["name"])
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as file:
        file.write(bytes(content, "utf8"))
    webbrowser.open("file://" + os.path.realpath(file.name))
    return file.name
