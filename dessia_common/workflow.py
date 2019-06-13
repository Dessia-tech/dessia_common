#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import inspect
import networkx as nx

class Model:
    def __init__(self, object_class, object_id=None):
        self.object_class = object_class

class Function:
    def __init__(self, function):
        self.function = function
        self.input_args = []
        for arg_name, parameter in inspect.signature(function).parameters.items():
            self.input_args.append(InputFunctionVariable(self, arg_name))
        

class Pipe:
    def __init__(self,
                 block1, variable_name1,
                 block2, variable_name2):
        self.block1 = block1
        self.block2 = block2
        self.variable_name1 = variable_name1
        self.variable_name2 = variable_name2

class AutomaticSort:
    def __init__(self):
        pass

class MileStone:
    def __init__(self):
        pass

class WorkFlow:
    def __init__(self, models, functions, pipes):
        self.models = models
        self.functions = functions
        self.pipes = pipes
        
    def graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.models)
        graph.add_nodes_from(self.functions)
        for function in self.functions:
            for input_parameter in function.input_args:
                graph.add_node(input_parameter)
                graph.add_edge(input_parameter, function)
        return graph
        
    def plot_graph(self):
        
        graph = self.graph()
        pos = nx.kamada_kawai_layout(graph)
        nx.draw_networkx_nodes(graph, pos, self.functions, node_shape='s')
        nx.draw_networkx_edges(graph, pos)

        labels = {f: f.function.__name__ for f in self.functions}
        for function in self.functions:
            for input_paramter in function.input_args:
                labels[input_paramter] = input_paramter.arg_name
        nx.draw_networkx_labels(graph, pos, labels)

class InputFunctionVariable:
    def __init__(self, function, arg_name):
        self.function = function
        self.arg_name = arg_name

        self.value = None

class OutputFunctionVariable:
    def __init__(self, function):
        self.function = function
        
        self.value = None


class ModelAttribute:
    def __init__(self, model, attribute_name):
        self.model = model
        self.attribute_name = attribute_name
    
        
import math

def sinus_f(x):
    return math.sin(x)

sinus = Function(sinus_f)

workflow = WorkFlow([], [sinus], [])

workflow.plot_graph()