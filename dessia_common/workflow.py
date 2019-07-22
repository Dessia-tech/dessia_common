#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#import os
import inspect
import time
import networkx as nx
import tempfile
import pkg_resources

from jinja2 import Environment, PackageLoader, select_autoescape
import webbrowser


class Variable:
    def __init__(self, name):
        self.name = name
        
    def copy(self):
        return Variable(self.name)
        
class VariableWithDefaultValue(Variable):
    def __init__(self, name, default_value):
        Variable.__init__(self, name)
        self.default_value = default_value
        
    def copy(self):
        return VariableWithDefaultValue(self.name, self.default_value)

class Block:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        

class InstanciateModel:
    def __init__(self, object_class):
        self.object_class = object_class
        
        inputs = []
        for arg_name, parameter in inspect.signature(self.object_class.__init__).parameters.items():
            if arg_name != 'self':
                inputs.append(Variable(arg_name))
        outputs = [Variable('Instanciated object')]
        Block.__init__(self, inputs, outputs)
        
    def evaluate(self, values):
        args = {var.name: values[var] for var in self.inputs}
        return [self.object_class(**args)]


class ModelMethod(Block):
    def __init__(self, model_class, method_name):
        self.model_class = model_class
        self.method_name = method_name
        inputs = [Variable('model at input')]
#        for arg_name, parameter in inspect.signature(getattr(self.model_class,
#                                                             self.method_name)).parameters.items():
#            if arg_name != 'self':
#                inputs.append(Variable(arg_name))
                
                
                
        args_specs = inspect.getfullargspec(getattr(self.model_class, self.method_name))

        nargs = len(args_specs.args) - 1
        
        if args_specs.defaults is not None:
            ndefault_args = len(args_specs.defaults)
        else:
            ndefault_args = 0
        
        for iargument, argument in enumerate(args_specs.args[1:]):
            if not argument in ['self']:
                if iargument >= nargs - ndefault_args:
#                    arguments.append((argument, args_specs.defaults[ndefault_args-nargs+iargument]))
                    inputs.append(VariableWithDefaultValue(argument,
                                                           args_specs.defaults[ndefault_args-nargs+iargument]))

                else: 
                    inputs.append(Variable(argument))
            
        outputs = [Variable('method result of {}'.format(self.method_name)),
                   Variable('model at output {}'.format(self.method_name))]
        Block.__init__(self, inputs, outputs)
        
    def evaluate(self, values):
        args = {var.name: values[var] for var in self.inputs[1:] if var in values}
        return [getattr(values[self.inputs[0]], self.method_name)(**args),
                values[self.inputs[0]]]
        
class Function(Block):
    def __init__(self, function):
        self.function = function
        inputs = []
        for arg_name, parameter in inspect.signature(function).parameters.items():
            inputs.append(Variable(arg_name))
        outputs = [Variable('Output function')]
        
        Block.__init__(self, inputs, outputs)
        
    def evaluate(self, values):
        return self.function(*values)

        
class ForEach(Block):
    def __init__(self, workflow, workflow_iterable_input):
        self.workflow = workflow
        self.workflow_iterable_input = workflow_iterable_input
        inputs = []
        for workflow_input in self.workflow.inputs:
            if workflow_input == workflow_iterable_input:
                inputs.append(Variable('Iterable input: '+workflow_input.name))
            else:
                input_ = workflow_input.copy()
                input_.name = 'binding '+input_.name
                inputs.append(input_)
                
#        input_iterable_variable = Variable('Foreach input iterable')
#        input_variables = [input_iterable_variable] + self.workflow.inputs
        output_variable = Variable('Foreach output')
        
        Block.__init__(self, inputs, [output_variable])

    def evaluate(self, values):
        values_workflow = {var2: values[var1] for var1, var2 in zip(self.inputs,
                                                                    self.workflow.inputs)}
        output_values = []
        for value in values_workflow[self.workflow_iterable_input]:
            values_workflow2 = {var.name: val for var, val in values.items() if var != self.workflow_iterable_input}
            values_workflow2[self.workflow_iterable_input] = value
            workflow_run = self.workflow.run(values_workflow2)
            output_values.append(workflow_run.output_value)
        return [output_values]
            

class ModelAttribute:
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
        
        Block.__init__(self, [Variable('Model')], [Variable('Model attribute')])

    def evaluate(self, values):
        return [getattr(values[self.inputs[0]], self.attribute_name)]

class Pipe:
    def __init__(self,
                 input_variable,
                 output_variable):
        self.input_variable = input_variable
        self.output_variable = output_variable



class WorkFlow(Block):
    def __init__(self, blocks, pipes, output):
        self.blocks = blocks
        self.pipes = pipes
        
        self.variables = []
        for block in self.blocks:
            self.variables.extend(block.inputs)
            self.variables.extend(block.outputs)
            
        self._utd_graph = False

        input_variables = []
        
        for variable in self.variables:
            if len(nx.ancestors(self.graph, variable)) == 0:
                input_variables.append(variable)

        Block.__init__(self, input_variables, [output])
                
    def _get_graph(self):
        if not self._utd_graph:        
            self._cached_graph = self._graph()
            self._utd_graph = True
        return self._cached_graph
            
    graph = property(_get_graph)
    
    def _graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.variables)
        graph.add_nodes_from(self.blocks)
        for block in self.blocks:
            for input_parameter in block.inputs:
                graph.add_edge(input_parameter, block)
            for output_parameter in block.outputs:
                graph.add_edge(block, output_parameter)
                
        for pipe in self.pipes:
            graph.add_edge(pipe.input_variable, pipe.output_variable)
        return graph
        
    def block_indices_connected_by_pipe(self, pipe):
        for iblock, block in enumerate(self.blocks):
            if pipe.input_variable in block.inputs:
                ib1 = iblock
                ti1 = 0
                iv1 = block.inputs.index(pipe.input_variable)
            if pipe.input_variable in block.outputs:
                ib1 = iblock
                ti1 = 1
                iv1 = block.outputs.index(pipe.input_variable)

            if pipe.output_variable in block.inputs:
                ib2 = iblock
                ti2 = 0
                iv2 = block.inputs.index(pipe.output_variable)
            if pipe.output_variable in block.outputs:
                ib2 = iblock
                ti2 = 1
                iv2 = block.outputs.index(pipe.output_variable)
            
        return (ib1, ti1, iv1), (ib2, ti2, iv2)
    
    def plot_graph(self):
        
        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, self.blocks,
                               node_shape='s', node_color='grey')
        nx.draw_networkx_nodes(self.graph, pos, self.variables, node_color='b')
        nx.draw_networkx_nodes(self.graph, pos, self.inputs, node_color='g')
        nx.draw_networkx_nodes(self.graph, pos, self.outputs, node_color='r')
        nx.draw_networkx_edges(self.graph, pos)
        

        labels = {}#b: b.function.__name__ for b in self.block}
        for block in self.blocks:
            labels[block] = block.__class__.__name__
            for variable in self.variables:
                labels[variable] = variable.name
#            labels[function.output] = 'Output function'
        nx.draw_networkx_labels(self.graph, pos, labels)


    def run(self, input_variables_values, verbose=False):
        log = ''
        activated_items = {p: False for p in self.pipes}
        activated_items.update({v: False for v in self.variables})
        activated_items.update({b: False for b in self.blocks})
            
        values = {}
        
#        for input_value, variable in zip(input_variables_values,
#                                         self.inputs):
        for variable in self.inputs:
            if variable in input_variables_values:                
                values[variable] = input_variables_values[variable]
                activated_items[variable] = True
            elif hasattr(variable, 'default_value'):
                values[variable] = variable.default_value
                activated_items[variable] = True
            else:
                print('Variable {} has no value or default value'.format(variable.name))
                raise ValueError
        
        something_activated = True
        
        start_time = time.time()
        
        log_line = 'Starting workflow run at {}'.format(time.strftime('%d/%m/%Y %H:%M:%S UTC',
                                                                      time.gmtime(start_time)))
        log += (log_line + '\n')
        if verbose:
            print(log_line)
        
        while something_activated:
            something_activated = False
            
            for pipe in self.pipes:
                if not activated_items[pipe]:
                    if activated_items[pipe.input_variable]:
                        activated_items[pipe] = True
                        values[pipe.output_variable] = values[pipe.input_variable]
                        activated_items[pipe.output_variable] = True
                        something_activated = True
            
            for block in self.blocks:
                if not activated_items[block]:
                    all_inputs_activated = True
                    for function_input in block.inputs:
                        
                        if not activated_items[function_input]:
                            all_inputs_activated = False
                            break
                        
                    if all_inputs_activated:
                        if verbose:
                            log_line = 'Evaluating block {}'.format(block.__class__.__name__)
                            log += log_line + '\n'
                            if verbose:
                                print(log_line)
                        output_values = block.evaluate({i: values[i]\
                                                        for i in block.inputs})
                        for output, output_value in zip(block.outputs, output_values):                            
                            values[output] = output_value
                            activated_items[output] = True
                        
                        activated_items[block] = True
                        something_activated = True
                       
        end_time = time.time()
        log_line = 'Workflow terminated in {} s'.format(end_time - start_time)
        
        log += log_line + '\n'
        if verbose:
            print(log_line)
        return WorkflowRun(self, values, start_time, end_time, log)
         
    def plot(self):
        env = Environment(loader=PackageLoader('dessia_common', 'templates'),
                          autoescape=select_autoescape(['html', 'xml']))


        template = env.get_template('workflow.html')

#        temp_folder = tempfile.mkdtemp()
        
        mx_path = pkg_resources.resource_filename(pkg_resources.Requirement('dessia_common'),
                                                  'dessia_common/templates/mxgraph')
        
        nodes = []
        for block in self.blocks:
            nodes.append({'name': block.__class__.__name__,
                          'inputs': [{'name': i.name,
                                      'workflow_input': i in self.inputs,
                                      'has_default': hasattr(block, 'default')}\
                                     for i in block.inputs],
                          'outputs': [{'name': o.name,
                                       'workflow_output': o in self.outputs}\
                                      for o in block.outputs]})
            
        edges = []
        for pipe in self.pipes:
#            (ib1, t1, ip1), (ib2, t2, ip2) = self.block_indices_connected_by_pipe
            edges.append(self.block_indices_connected_by_pipe(pipe))
        
        options = {}
        s = template.render(
            mx_path=mx_path,
            nodes=nodes,
            edges=edges,
            options=options)

        temp_file = tempfile.mkstemp(suffix='.html')[1]
#        print(temp_file, nodes)
        
        with open(temp_file, 'wb') as file:
            file.write(s.encode('utf-8'))

        webbrowser.open('file://' + temp_file)
                            
class WorkflowRun:
    def __init__(self, workflow, values, start_time, end_time, log):
        self.workflow = workflow
        self.values = values
        
        self.output_value = self.values[self.workflow.outputs[0]]
        
        self.start_time = start_time
        self.end_time = end_time
        self.execution_time = end_time - start_time
        self.log = log
    
