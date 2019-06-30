#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import inspect
import networkx as nx

class Model:
    def __init__(self, object_class, object_id=None):
        self.object_class = object_class
        self.object_id = object_id

class InstanciateModel:
    def __init__(self, object_class):
        self.object_class = object_class



class ModelMethod:
    def __init__(self, model, method_name):
        self.model = model
        self.method_name = method_name
        
class Function:
    def __init__(self, function):
        self.function = function
        self.input_args = []
        for arg_name, parameter in inspect.signature(function).parameters.items():
            self.input_args.append(InputFunctionVariable(self, arg_name))
        self.output = OutputFunctionVariable(self)
        
    def evaluate(self, values):
        return self.function(*values)

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

class Pipe:
    def __init__(self,
                 input_variable,
                 output_variable):
        self.input_variable = input_variable
        self.output_variable = output_variable


class WorkFlow:
    def __init__(self, blocks, pipes):
        self.blocks = blocks
        self.pipes = pipes
        
        self.variables = []
        for block in self.blocks:
            self.variables.extend(function.input_args)
            self.variables.append(function.output)
            
        self._utd_graph = False

        self.input_variables = []
        
        for variable in self.variables:
            if len(nx.ancestors(self.graph, variable)) == 0:
                self.input_variables.append(variable)

                
    def _get_graph(self):
        if not self._utd_graph:        
            self._cached_graph = self._graph()
            self._utd_graph = True
        return self._cached_graph
            
    graph = property(_get_graph)
    
    def _graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.models)
        graph.add_nodes_from(self.functions)
        for function in self.functions:
            for input_parameter in function.input_args:
                graph.add_node(input_parameter)
                graph.add_edge(input_parameter, function)
            graph.add_node(function.output)
            graph.add_edge(function, function.output)
        for pipe in self.pipes:
            graph.add_edge(pipe.input_variable, pipe.output_variable)
        return graph
        
    
    def plot_graph(self):
        
        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, self.functions,
                               node_shape='s', node_color='grey')
        nx.draw_networkx_nodes(self.graph, pos, self.variables)
        nx.draw_networkx_nodes(self.graph, pos, self.input_variables, node_color='r')
        nx.draw_networkx_edges(self.graph, pos)
        

        labels = {f: f.function.__name__ for f in self.functions}
        for function in self.functions:
            for input_paramter in function.input_args:
                labels[input_paramter] = input_paramter.arg_name
            labels[function.output] = 'Output function'
        nx.draw_networkx_labels(self.graph, pos, labels)


    def run(self, input_variables_values):
        activated_items = {p: False for p in self.pipes}
        activated_items.update({v: False for v in self.variables})
        activated_items.update({f: False for f in self.functions})
        activated_items.update({m: False for m in self.models})

        if len(input_variables_values) != len(self.input_variables):
            raise ValueError
            
        values = {}
        
        for input_value, variable in zip(input_variables_values,
                                         self.input_variables):
            values[variable] = input_value
            activated_items[variable] = True
        
        something_activated = True
        
        while something_activated:
            something_activated = False
            
            for pipe in self.pipes:
                if not activated_items[pipe]:
                    if activated_items[pipe.input_variable]:
                        activated_items[pipe] = True
                        values[pipe.output_variable] = values[pipe.input_variable]
                        activated_items[pipe.output_variable] = True
                        something_activated = True
            
            for function in self.functions:
                if not activated_items[function]:
                    all_inputs_activated = True
                    for function_input in function.input_args:
                        
                        if not activated_items[function_input]:
                            all_inputs_activated = False
                            break
                        
                    if all_inputs_activated:
                        output_value = function.evaluate([values[i]\
                                                          for i in function.input_args])
                        values[function.output] = output_value
                        activated_items[function] = True
                        activated_items[function.output] = True
                        something_activated = True
                        
        return WorkflowRun(self, values)
            
                            
class WorkflowRun:
    def __init__(self, workflow, values):
        self.workflow = workflow
        self.values = values
        
