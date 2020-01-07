#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import inspect
import time
import tempfile
from importlib import import_module
import webbrowser
import networkx as nx
import pkg_resources
import typing

from jinja2 import Environment, PackageLoader, select_autoescape
import dessia_common as dc


class Variable(dc.DessiaObject):
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Variable",
        "required": ["_name"],
        "properties": {
            "_name": {
                "type": "string",
                "editable": True,
                }
            }
        }
    _standalone_in_db = False
    def __init__(self, _name, name=''):
        self._name = _name

        dc.DessiaObject.__init__(self, name=name)

    def equivalent_hash(self):
        return int(hash(self._name) % 10e5)

    def equivalent(self, other_variable):
        if self.__class__ != other_variable.__class__:
            return False
        return self._name == other_variable._name

    def copy(self):
        return Variable(self._name, name=self.name)

class TypedVariable(Variable):
    def __init__(self, _name, type_, name=''):
        Variable.__init__(self, _name, name=name)
        self.type_ = type_

    def __getstate__(self):
        dict_ = self.__dict__.copy()
        if hasattr(self.type_, '_name') and self.type_._name == 'List':
            arg = self.type_.__args__[0]
            if arg.__module__ == 'builtins':
                full_argname = '__builtins__.' + arg.__name__
            else:
                full_argname = arg.__module__ + '.' + arg.__name__
            dict_['type_'] = 'List[' + full_argname + ']'
        return dict_

    def __setstate__(self, dict_):
        if isinstance(dict_['type_'], str):
            splitted_type = dict_['type_'].split('[')
            if splitted_type[0] == 'List':
                full_argname = splitted_type[1].split(']')[0]
                splitted_argname = full_argname.rsplit('.', 1)
                if splitted_argname[0] != '__builtins__':
                    exec('import ' + splitted_argname[0])
                    type_ = eval(full_argname)
                else:
                    type_ = eval(splitted_argname[1])
                self.type_ = typing.List[type_]

    def equivalent_hash(self):
        return int((hash(self._name) + hash(self.type_.__name__)) % 10e5)

    def equivalent(self, other_variable):
        if self.__class__ != other_variable.__class__:
            return False
        return self._name == other_variable._name and self.type_ == other_variable.type_

    def copy(self):
        return TypedVariable(self._name, self.type_, name=self.name)

class VariableWithDefaultValue(Variable):
    def __init__(self, _name, default_value, name=''):
        Variable.__init__(self, _name, name='')
        self.default_value = default_value

    def equivalent_hash(self):
        return int((hash(self._name) + hash(self.default_value)) % 10e5)

    def equivalent(self, other_variable):
        if self.__class__ != other_variable.__class__:
            return False
        return self._name == other_variable._name and self.default_value == other_variable.default_value

    def copy(self):
        return VariableWithDefaultValue(self._name, self.default_value, name=self.name)

class TypedVariableWithDefaultValue(TypedVariable):
    def __init__(self, _name, type_, default_value, name=''):
        TypedVariable.__init__(self, _name, type_, name=name)
        self.default_value = default_value

    def equivalent_hash(self):
        return int((hash(self._name)\
                    + hash(self.type_.__name__)\
                    + hash(self.default_value)) % 10e5)

    def equivalent(self, other_variable):
        if self.__class__ != other_variable.__class__:
            return False
        return (self._name == other_variable._name\
                and self.type_ == other_variable.type_\
                and self.default_value == other_variable.default_value)

    def copy(self):
        return TypedVariableWithDefaultValue(self._name, self.type_, self.default_value, name=self.name)

class Block(dc.DessiaObject):
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Block",
        "required": ["inputs", "outputs"],
        "properties": {
            "inputs": {
                "type": "array",
                "editable": False,
                "items" : {
                    "type" : "object",
                    "classes" : ["dessia_common.workflow.Variable",
                                 "dessia_common.workflow.TypedVariable"
                                 "dessia_common.workflow.VariableWithDefaultValue",
                                 "dessia_common.workflow.TypedVariableWithDefaultValue"],
                    "editable" : False,
                    },
                },
            "outputs": {
                "type": "array",
                "editable": False,
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        "classes" : ["dessia_common.workflow.Variable",
                                     "dessia_common.workflow.TypedVariable"
                                     "dessia_common.workflow.VariableWithDefaultValue",
                                     "dessia_common.workflow.TypedVariableWithDefaultValue"],
                        "editable" : False
                        },
                    }
                }
            }
        }
    _standalone_in_db = False

    def __init__(self, inputs, outputs, name=''):
        self.inputs = inputs
        self.outputs = outputs

        dc.DessiaObject.__init__(self, name=name)

    def equivalent_hash(self):
        return len(self.__class__.__name__)


    def equivalent(self, other_block):
        if not self.__class__.__name__ == other_block.__class__.__name__:
            return False
        return True

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['block_class'] = self.__class__.__name__
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        if dict_['block_class'] in ['InstanciateModel', 'ModelMethod',
                                    'ForEach', 'Function', 'ModelAttribute',
                                    'Filter']:
            return eval(dict_['block_class']).dict_to_object(dict_)

    @property
    def _method_jsonschemas(self):
        jsonschemas = {'run': dc.JSONSCHEMA_HEADER}
        properties_dict = jsonschemas['run']['properties']
        required_inputs = []
        for i, input_ in enumerate(self.inputs):
            current_dict = {}
            annotation = (str(i), input_.type_)

            annotation_jsonschema = dc.jsonschema_from_annotation(annotation=annotation,
                                                                  jsonschema_element=current_dict,
                                                                  order=i,
                                                                  title=dc.prettyname(input_.name))
            current_dict.update(annotation_jsonschema[str(i)])
            if not isinstance(input_, (VariableWithDefaultValue, TypedVariableWithDefaultValue)):
                required_inputs.append(str(i))
            else:
                current_dict.update(dc.set_default_value(current_dict,
                                                         str(i),
                                                         input_.default_value))
            properties_dict[str(i)] = current_dict[str(i)]
        jsonschemas['run']['required'] = required_inputs
        return jsonschemas


class InstanciateModel(Block):
    _jsonschema = dc.dict_merge(Block._jsonschema, {
        "title" : "Instantiate model Base Schema",
        "required": ['object_class'],
        "properties": {
            "object_class": {
                "type" : "string",
                "editable" : True,
                "examples" : ['Nom']
                }
            }
        })
    def __init__(self, object_class, name=''):
        self.object_class = object_class
        inputs = []

        inputs = set_inputs(self.object_class.__init__, inputs, name)

        outputs = [TypedVariable('Instanciated object', self.object_class)]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.object_class.__name__)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.object_class.__class__.__name__ == other_block.object_class.__class__.__name__

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'object_class': self.object_class.__name__,
                      'object_class_module': self.object_class.__module__})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        # TODO: Eval is dangerous: check that it is a class before
        class_ = getattr(import_module(dict_['object_class_module']),
                         dict_['object_class'])
        return cls(class_, name=dict_['name'])


    def evaluate(self, values):
        args = {var._name: values[var] for var in self.inputs}
        return [self.object_class(**args)]


class ModelMethod(Block):
    _jsonschema = dc.dict_merge(Block._jsonschema, {
        "title" : "Model method Base Schema",
        "required": ['model_class', 'method_name'],
        "properties": {
            "model_class": {
                "type" : "string",
                "editable" : True,
                "examples" : ['Nom']
                },

            "method_name": {
                "type" : "string",
                "editable" : True,
                "examples" : ['Nom']
                }
            }
        })
    def __init__(self, model_class, method_name, name=''):
        self.model_class = model_class
        self.method_name = method_name
        inputs = [TypedVariable('model at input', model_class)]
        method = getattr(self.model_class, self.method_name)

        inputs = set_inputs(method, inputs, name)

        outputs = [TypedVariable('method result of {}'.format(self.method_name), method.__annotations__['return']),
                   TypedVariable('model at output {}'.format(self.method_name), model_class)]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.model_class.__name__) + 7*len(self.method_name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.model_class.__name__ == other_block.model_class.__name__\
               and self.method_name == other_block.method_name


    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'model_class': self.model_class.__name__,
                      'model_module': self.model_class.__module__,
                      'method_name': self.method_name})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        class_ = getattr(import_module(dict_['model_module']),
                         dict_['model_class'])
        return cls(class_, dict_['method_name'], dict_['name'])


    def evaluate(self, values):
        args = {var._name: values[var] for var in self.inputs[1:] if var in values}
        return [getattr(values[self.inputs[0]], self.method_name)(**args),
                values[self.inputs[0]]]

class Function(Block):
    def __init__(self, function, name=''):
        self.function = function
        inputs = []
        for arg_name in inspect.signature(function).parameters.keys():
            inputs.append(TypedVariable(arg_name, function.__annotations__[arg_name]))
        outputs = [TypedVariable('Output function', function.__annotations__['return'])]

        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return int(hash(self.function.__name__) % 10e5)

    def equivalent(self, other_block):
        return self.method == other_block.method

    def evaluate(self, values):
        return self.function(*values)


class ForEach(Block):

    def __init__(self, workflow, workflow_iterable_input, name=''):
        self.workflow = workflow
        self.workflow_iterable_input = workflow_iterable_input
        inputs = []
        for workflow_input in self.workflow.inputs:
            if workflow_input == workflow_iterable_input:
                inputs.append(Variable('Iterable input: '+workflow_input._name))
            else:
                input_ = workflow_input.copy()
                input_._name = 'binding '+input_._name
                inputs.append(input_)
        output_variable = Variable('Foreach output')

        Block.__init__(self, inputs, [output_variable], name=name)

    def equivalent_hash(self):
        return int(hash(self.workflow) % 10e5)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.workflow == other_block.workflow\
               and self.workflow.variable_indices(self.workflow_iterable_input)\
                   == other_block.workflow.variable_indices(other_block.workflow_iterable_input)

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({
            'workflow': self.workflow.to_dict(),
            'workflow_iterable_input': self.workflow.variable_indices(self.workflow_iterable_input)
            })
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        workflow = Workflow.dict_to_object(dict_['workflow'])
        ib, _, ip = dict_['workflow_iterable_input']

        workflow_iterable_input = workflow.blocks[ib].inputs[ip]
        return cls(workflow, workflow_iterable_input)

    def evaluate(self, values):
        values_workflow = {var2: values[var1] for var1, var2 in zip(self.inputs,
                                                                    self.workflow.inputs)}
        output_values = []
        for value in values_workflow[self.workflow_iterable_input]:
            values_workflow2 = {var._name: val\
                                for var, val in values.items()\
                                if var != self.workflow_iterable_input}
            values_workflow2[self.workflow_iterable_input] = value
            workflow_run = self.workflow.run(values_workflow2)
            output_values.append(workflow_run.output_value)
        return [output_values]

class Filter(Block):
    def __init__(self, filters, name=''):
        self.filters = filters
        inputs = [Variable('input_list')]
        outputs = [Variable('output_list')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.filters == other_block.filters

    def equivalent_hash(self):
        return int(sum([hash(v) for f in self.filters for v in f.values()]) % 10e5)

    def _display_angular(self):
        displays = [{'angular_component': 'results',
                     'filters': self.filters}]
        return displays

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'filters': self.filters})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        return cls(dict_['filters'], dict_['name'])

    def evaluate(self, values):
        ouput_values = []
        objects = values[self.inputs[0]]
        for object_ in objects:
            valid = True
            for filter_ in self.filters:
                attribute_path = filter_['attribute']
                operator = filter_['operator']
                bound = filter_['bound']
                attribute = dc.getdeepattr(object_, attribute_path)
                if operator == 'lte' and attribute > bound:
                    valid = False
                if operator == 'gte' and attribute < bound:
                    valid = False

                if operator == 'lt' and attribute >= bound:
                    valid = False
                if operator == 'gt' and attribute <= bound:
                    valid = False

                if operator == 'eq' and attribute != bound:
                    valid = False

            if valid:
                ouput_values.append(object_)
        return [ouput_values]


class ModelAttribute(Block):
    def __init__(self, attribute_name, name=''):
        self.attribute_name = attribute_name

        Block.__init__(self, [Variable('Model')], [Variable('Model attribute')], name=name)

    def equivalent_hash(self):
        return len(self.attribute_name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.attribute_name == other_block.attribute_name


    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'attribute_name': self.attribute_name})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        return cls(dict_['attribute_name'], dict_['name'])

    def evaluate(self, values):
        return [getattr(values[self.inputs[0]], self.attribute_name)]

class Pipe(dc.DessiaObject):
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Block",
        "required": ["input_variable", "output_variable"],
        "properties": {
            "input_variable": {
                "type": "object",
                "editable": True,
                "classes" : ["dessia_common.workflow.Variable",
                             "dessia_common.workflow.TypedVariable"
                             "dessia_common.workflow.VariableWithDefaultValue",
                             "dessia_common.workflow.TypedVariableWithDefaultValue"],
                },
            "output_variable": {
                "type": "object",
                "editable": True,
                "classes" : ["dessia_common.workflow.Variable",
                             "dessia_common.workflow.TypedVariable"
                             "dessia_common.workflow.VariableWithDefaultValue",
                             "dessia_common.workflow.TypedVariableWithDefaultValue"],
                }
            }
        }
    def __init__(self,
                 input_variable,
                 output_variable):
        self.input_variable = input_variable
        self.output_variable = output_variable


    def to_dict(self):
        return {'input_variable': self.input_variable,
                'output_variable': self.output_variable}


class Workflow(Block):
    _standalone_in_db = True

    _dessia_methods = ['run']

    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Workflow",
        "required": ["blocks",
                     "pipes",
                     "outputs"],
        "properties": {
            "blocks": {
                "type": "array",
                "order": 0,
                "editable" : True,
                "items" : {
                    "type" : "object",
                    "classes" : ["dessia_common.workflow.InstanciateModel",
                                 "dessia_common.workflow.ModelMethod",
                                 "dessia_common.workflow.ForEach",
                                 "dessia_common.workflow.ModelAttribute"],
                    "editable" : True,
                    },
                },
            "pipes": {
                "type": "array",
                "order": 1,
                "editable" : True,
                "items": {
                    'type': 'objects',
                    'classes' : ["dessia_common.workflow.Pipe"],
                    "editable": True
                    }
                },
            "outputs": {
                "type": "array",
                "order": 2,
                'items': {
                    'type': 'array',
                    'items': {'type': 'number'}
                    }
                }
            }
        }


    def __init__(self, blocks, pipes, output, name=''):
        self.blocks = blocks
        self.pipes = pipes

        self.coordinates = {}

        self.variables = []
        for block in self.blocks:
            self.variables.extend(block.inputs)
            self.variables.extend(block.outputs)
            self.coordinates[block] = (0, 0)

        self._utd_graph = False

        input_variables = []

        for variable in self.variables:
            if len(nx.ancestors(self.graph, variable)) == 0: # !!! Why not just : if nx.ancestors(self.graph, variable) ?
                input_variables.append(variable)

        Block.__init__(self, input_variables, [output], name=name)

    def __hash__(self):
        base_hash = len(self.blocks)+11*len(self.pipes)+sum(self.variable_indices(self.outputs[0]))
        block_hash = int(sum([b.equivalent_hash() for b in self.blocks]) % 10e5)
        return base_hash + block_hash

    def __eq__(self, other_workflow):
        if hash(self) != hash(other_workflow):
            return False

        if not Block.equivalent(self, other_workflow):
            return False

        # TODO: temp , reuse graph!!!!
        for block1, block2 in zip(self.blocks, other_workflow.blocks):
            if not block1.equivalent(block2):
                return False

        return True

#        graph_matcher = nx.algorithms.isomorphism.GraphMatcher(self.graph,
#                                                               other_workflow.graph,
#                                                               node_match=node_matcher)
#
#        isomorphic = graph_matcher.is_isomorphic()
#        if isomorphic:
#            for mapping in graph_matcher.isomorphisms_iter():
#                mapping_valid = True
#                for element1, element2 in mapping.items():
#                    if not isinstance(element1, Variable) and (element1 != element2):
#                        mapping_valid = False
#                        break
#                if mapping_valid:
#                    if mapping[self.outputs[0]] == other_workflow.outputs[0]:
#                        return True
#            return False
#        return False

    def _display_angular(self):
        displays = []
        nodes, edges = self.jointjs_data()
        displays.extend([{'angular_component': 'workflow',
                         'nodes': nodes,
                         'edges': edges}])
        return displays

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        blocks = [b.to_dict() for b in self.blocks]
        pipes = []
        for pipe in self.pipes:
            pipes.append((self.variable_indices(pipe.input_variable),
                          self.variable_indices(pipe.output_variable)))

        dict_.update({'blocks': blocks,
                      'pipes': pipes,
                      'output': self.variable_indices(self.outputs[0])})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        blocks = [Block.dict_to_object(d) for d in dict_['blocks']]
        pipes = []
        for (ib1, _, ip1), (ib2, _, ip2) in dict_['pipes']:
            variable1 = blocks[ib1].outputs[ip1]
            variable2 = blocks[ib2].inputs[ip2]
            pipes.append(Pipe(variable1, variable2))

        output = blocks[dict_['output'][0]].outputs[dict_['output'][2]]

        return cls(blocks, pipes, output, name=dict_['name'])

    def dict_to_arguments(self, dict_, method):
        arguments_values = {}
        for i, input_ in enumerate(self.inputs):
            if not isinstance(input_, (VariableWithDefaultValue, TypedVariableWithDefaultValue))\
            or (isinstance(input_, (VariableWithDefaultValue, TypedVariableWithDefaultValue))
                and str(i) in dict_):
                value = dict_[str(i)]
                deserialized_value = dc.deserialize_argument(input_.type_, value)
                arguments_values[i] = deserialized_value

        arguments = {'input_variables_values': arguments_values}
        return arguments

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

    def variable_indices(self, variable):
        for iblock, block in enumerate(self.blocks):
            if variable in block.inputs:
                ib1 = iblock
                ti1 = 0
                iv1 = block.inputs.index(variable)
            if variable in block.outputs:
                ib1 = iblock
                ti1 = 1
                iv1 = block.outputs.index(variable)

        return (ib1, ti1, iv1)

    def index(self, variable):
        index = self.inputs.index(variable)
        return index

    def layout(self, n_x_anchors):
        for iblock, block in enumerate(self.blocks):
            self.coordinates[block] = (iblock % n_x_anchors, iblock // n_x_anchors)


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
                labels[variable] = variable._name
        nx.draw_networkx_labels(self.graph, pos, labels)


    def run(self, input_variables_values, verbose=False):
        log = ''
        activated_items = {p: False for p in self.pipes}
        activated_items.update({v: False for v in self.variables})
        activated_items.update({b: False for b in self.blocks})

        values = {}
        for index, variable in enumerate(self.inputs):
            if index in input_variables_values:
                values[variable] = input_variables_values[index]
                activated_items[variable] = True
            elif hasattr(variable, 'default_value'):
                values[variable] = variable.default_value
                activated_items[variable] = True
            else:
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

#        workflow_run_values = [values[variable] for variable in self.variables]
#        self.variables.index(self.outputs[0])
        output_value = values[self.outputs[0]]
        return WorkflowRun(self, output_value, start_time, end_time, log)

    def mxgraph_data(self):
        nodes = []
        for block in self.blocks:
            nodes.append({'name': block.__class__.__name__,
                          'inputs': [{'name': i._name,
                                      'workflow_input': i in self.inputs,
                                      'has_default': hasattr(block, 'default')}\
                                     for i in block.inputs],
                          'outputs': [{'name': o._name,
                                       'workflow_output': o in self.outputs}\
                                      for o in block.outputs]})

        edges = []
        for pipe in self.pipes:
            edges.append((self.variable_indices(pipe.input_variable),
                          self.variable_indices(pipe.output_variable)))

        return nodes, edges


    def jointjs_data(self):
        nodes = []
        for block in self.blocks:
            nodes.append({'name': block.__class__.__name__,
                          'inputs': [i._name for i in block.inputs],
                          'outputs': [o._name for o in block.outputs]})

        edges = []
        for pipe in self.pipes:
            ib1, is1, ip1 = self.variable_indices(pipe.input_variable)
            if is1:
                block = self.blocks[ib1]
                ip1 += len(block.inputs)

            ib2, is2, ip2 = self.variable_indices(pipe.output_variable)
            if is2:
                block = self.blocks[ib2]
                ip2 += len(block.inputs)

            edges.append(((ib1, ip1), (ib2, ip2)))

        return nodes, edges


    def plot_mxgraph(self):
        env = Environment(loader=PackageLoader('dessia_common', 'templates'),
                          autoescape=select_autoescape(['html', 'xml']))


        template = env.get_template('workflow.html')

        mx_path = pkg_resources.resource_filename(pkg_resources.Requirement('dessia_common'),
                                                  'dessia_common/templates/mxgraph')

        nodes, edges = self.mxgraph_data()
        options = {}
        rendered_template = template.render(mx_path=mx_path,
                                            nodes=nodes,
                                            edges=edges,
                                            options=options)

        temp_file = tempfile.mkstemp(suffix='.html')[1]

        with open(temp_file, 'wb') as file:
            file.write(rendered_template.encode('utf-8'))

        webbrowser.open('file://' + temp_file)


    def plot_jointjs(self):
        env = Environment(loader=PackageLoader('dessia_common', 'templates'),
                          autoescape=select_autoescape(['html', 'xml']))


        template = env.get_template('workflow_jointjs.html')

        nodes, edges = self.jointjs_data()
        options = {}
        rendered_template = template.render(nodes=nodes,
                                            edges=edges,
                                            options=options)

        temp_file = tempfile.mkstemp(suffix='.html')[1]

        with open(temp_file, 'wb') as file:
            file.write(rendered_template.encode('utf-8'))

        webbrowser.open('file://' + temp_file)

    def find_variable_type(self, variable):
        for block in self.blocks:
            for block_variable in block.outputs: # !!! just ouputs is  a workaround
                if block_variable._name == variable._name and isinstance(block, ModelAttribute): # !!! isinstance is just a workaround
                    for var in self.inputs + self.outputs:
                        if var._name == block.attribute_name:
                            return var.type_
        return None


class WorkflowRun(dc.DessiaObject):
    _standalone_in_db = True
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "WorkflowRun Base Schema",
        "required": [],
        "properties": {
            "workflow" : {
                "type" : "object",
                "title" : "Workflow",
                "classes" : ["dessia_common.workflow.Workflow"],
                "order" : 0,
                "editable" : False,
                "description" : "Workflow"
                },
            'output_value': {
                "type" : "array",
                "items": {
                    "type" : "object",
                    "classes" : "*"
                    },
                "title" : "Values",
                "description" : "Input and output values",
                "editable" : False,
                "order" : 1
                },
            'start_time': {
                "type": "number",
                "title" : "Start Time",
                "editable": False,
                "description": "Start time of simulation",
                "order" : 2
                },
            'end_time': {
                "type": "number",
                "title" : "End Time",
                "editable": False,
                "description": "End time of simulation",
                "order" : 3
                 },
            'log': {
                "type": "string",
                "title" : "Log",
                "editable": False,
                "description": "Log",
                "order" : 4
                 }
             }
         }

    def __init__(self, workflow, output_value, start_time, end_time, log, name=''):
        self.workflow = workflow
#        self.values = values

#        output_index = workflow.variables.index(workflow.outputs[0])
        self.output_value = output_value
#        self.output_value = self.values[self.workflow.outputs[0]]

        self.start_time = start_time
        self.end_time = end_time
        self.execution_time = end_time - start_time
        self.log = log

        dc.DessiaObject.__init__(self, name=name)

    def __eq__(self, other_workflow_run):
        if hasattr(self.output_value, '__iter__'):
            equal_output = (hasattr(self.output_value, '__iter__')\
                            and all([v == other_v\
                                     for v, other_v\
                                     in zip(self.output_value,
                                            other_workflow_run.output_value)]))
        else:
            equal_output = self.output_value == other_workflow_run.output_value
        return self.workflow == other_workflow_run.workflow and equal_output

    def __hash__(self):
        if hasattr(self.output_value, '__iter__'):
            hash_output = int(sum([hash(v) for v in self.output_value]) % 10e5)
        else:
            hash_output = hash(self.output_value)
        return hash(self.workflow) + int(hash_output % 10e5)

    def _display_angular(self):
        displays = self.workflow._display_angular()

        for block in self.workflow.blocks:
            if isinstance(block, Filter):
                filter_display = block._display_angular()
                values = [{f['attribute'] : dc.getdeepattr(v, f['attribute'])\
                           for f in filter_display[0]['filters']}\
                          for v in self.output_value]
                filter_display[0]['values'] = values
                filter_display[0]['references_attribute'] = 'output_value'
                displays.extend(filter_display)
        return displays

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'workflow' : self.workflow.to_dict(),
                      'output_value' : dc.serialize_sequence(self.output_value),
                      'output_value_type' : dc.recursive_type(self.output_value),
                      'start_time' : self.start_time,
                      'end_time' : self.end_time,
                      'execution_time' : self.execution_time,
                      'log' : self.log})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        workflow = Workflow.dict_to_object(dict_['workflow'])
        output_value = dc.recursive_instantiation(dict_['output_value_type'], dict_['output_value'])
        return cls(workflow=workflow, output_value=output_value,
                   start_time=dict_['start_time'], end_time=dict_['end_time'],
                   log=dict_['log'], name=dict_['name'])

def set_inputs(method, inputs=[], block_name=''):
    args_specs = inspect.getfullargspec(method)
    nargs = len(args_specs.args) - 1

    if args_specs.defaults is not None:
        ndefault_args = len(args_specs.defaults)
    else:
        ndefault_args = 0

    for iargument, argument in enumerate(args_specs.args[1:]):
        if not argument in ['self', 'progress_callback']:
            if iargument >= nargs - ndefault_args:
                default_value = args_specs.defaults[ndefault_args-nargs+iargument]
                inputs.append(TypedVariableWithDefaultValue(_name=argument,
                                                            type_=method.__annotations__[argument],
                                                            default_value=default_value,
                                                            name=block_name + ' - ' + argument))

            else:
                inputs.append(TypedVariable(_name=argument,
                                            type_=method.__annotations__[argument],
                                            name=block_name + ' - ' + argument))
    return inputs

#def node_matcher(n1, n2):
#    if n1.__class__.__name__ != n2.__class__.__name__:
#        return False
#    else:
#        return n1.equivalent(n2)
