#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import inspect
import time
import tempfile
import json
from importlib import import_module
import webbrowser
import networkx as nx
# import pkg_resources
# import typing
from copy import deepcopy
import typeguard

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

    def __init__(self, name=''):
        dc.DessiaObject.__init__(self, name=name)

class TypedVariable(Variable):
    def __init__(self, type_, name=''):
        Variable.__init__(self, name=name)
        self.type_ = type_

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['type'] = dc.serialize_typing(self.type_)

        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        type_ = dc.get_python_class_from_class_name(dict_['type'])
        return cls(type_=type_, name=dict_['name'])


class VariableWithDefaultValue(Variable):
    def __init__(self, default_value, name=''):
        Variable.__init__(self, name=name)
        self.default_value = default_value

    

class TypedVariableWithDefaultValue(TypedVariable):
    def __init__(self, type_, default_value, name=''):
        TypedVariable.__init__(self, type_, name=name)
        self.default_value = default_value

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['type'] = dc.serialize_typing(self.type_)
        dict_['default_value'] = dc.serialize(self.default_value)
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        type_ = dc.get_python_class_from_class_name(dict_['type_'])
        default_value = dc.deserialize(dict_['default_value'])
        return cls(type_=type_, default_value=default_value, name=dict_['name'])

    def copy(self):
        return TypedVariableWithDefaultValue(self.type_, self.default_value, name=self.name)


def set_block_variable_names_from_dict(func):
   def func_wrapper(cls, dict_):
       obj = func(cls, dict_)
       if 'input_names' in dict_:
           for input_name, input_ in zip(dict_['input_names'], obj.inputs):
               input_.name = input_name
       if 'output_names' in dict_:
           for output_name, output_ in zip(dict_['output_names'], obj.outputs):
               output_.name = output_name
       return obj
   return func_wrapper

    

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
    _non_serializable_attributes = ['inputs', 'outputs']

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
        dict_['input_names'] = [i.name for i in self.inputs]
        dict_['output_names'] = [o.name for o in self.outputs]
        return dict_


    @property
    def _method_jsonschemas(self):
        jsonschemas = {'run': deepcopy(dc.JSONSCHEMA_HEADER)}
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
    
    def jointjs_data(self):
        data = {'block_class': self.__class__.__name__}
        if self.name != '':
            data['name'] = self.name
        else:
            data['name'] = self.__class__.__name__
                
        return data
    


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

    def __init__(self, model_class, name=''):
        self.model_class = model_class
        inputs = []

        inputs = set_inputs_from_function(self.model_class.__init__, inputs, name)

        outputs = [TypedVariable(self.model_class, 'Instanciated object')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.model_class.__name__)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.model_class.__class__.__name__ == other_block.model_class.__class__.__name__

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'model_class': self.model_class.__name__,
                      'model_class_module': self.model_class.__module__})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        # TODO: Eval is dangerous: check that it is a class before
        class_ = getattr(import_module(dict_['model_class_module']),
                          dict_['model_class'])
        return cls(class_, name=dict_['name'])


    def evaluate(self, values):
        args = {var.name: values[var] for var in self.inputs}
        return [self.model_class(**args)]
    
    
    def package_mix(self):
        return {self.model_class.__module__.split('.')[0]: 1}

class ClassMethod(Block):

    def __init__(self, class_, method_name, name=''):
        self.class_ = class_
        self.method_name = method_name
        inputs = []
        method = getattr(self.class_, self.method_name)
        inputs = set_inputs_from_function(method, inputs, name)
        
        self.argument_names = [i.name for i in inputs]

        outputs = [TypedVariable(dc.type_from_annotation(method.__annotations__['return'],
                                                         method.__module__),
                                 'method result of {}'.format(self.method_name))]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.class_.__name__) + 7*len(self.method_name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.class_.__name__ == other_block.class_.__name__\
               and self.method_name == other_block.method_name

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'method_name': self.method_name,
                      'class_': self.class_.__name__,
                      'class_module': self.class_.__module__})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        # TODO: Eval is dangerous: check that it is a class before
        class_ = getattr(import_module(dict_['class_module']),
                          dict_['class_'])
        return cls(class_=class_,
                   method_name=dict_['method_name'],
                   name=dict_['name'])


    def evaluate(self, values):
        args = {arg_name: values[var] for arg_name, var in zip(self.argument_names,
                                                               self.inputs) if var in values}
        return [getattr(self.class_, self.method_name)(**args)]


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
        inputs = [TypedVariable(model_class, 'model at input')]
        method = getattr(self.model_class, self.method_name)

        inputs = set_inputs_from_function(method, inputs, name)
        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        if 'return' in method.__annotations__:
            return_output = TypedVariable(dc.type_from_annotation(method.__annotations__['return'],
                                                                  method.__module__),
                                          'method result of {}'.format(self.method_name))
        else:
            return_output = Variable('method result of {}'.format(self.method_name))
            
        outputs = [return_output,
                   TypedVariable(model_class,
                                 'model at output {}'.format(self.method_name))]
        if name == '':
            name = 'Model method: {}'.format(self.method_name)
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.model_class.__name__) + 7*len(self.method_name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.model_class.__name__ == other_block.model_class.__name__\
               and self.method_name == other_block.method_name


    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'method_name': self.method_name,
                      'model_class': self.model_class.__name__,
                      'model_class_module': self.model_class.__module__})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        # TODO: Eval is dangerous: check that it is a class before
        class_ = getattr(import_module(dict_['model_class_module']),
                          dict_['model_class'])
        return cls(model_class=class_,
                   method_name=dict_['method_name'],
                   name=dict_['name'])

    def evaluate(self, values):
        args = {arg_name: values[var] for arg_name, var in zip(self.argument_names, 
                                                               self.inputs[1:]) if var in values}
        return [getattr(values[self.inputs[0]], self.method_name)(**args),
                values[self.inputs[0]]]

    def package_mix(self):
        return {self.model_class.__module__.split('.')[0]: 1}


class Function(Block):
    def __init__(self, function, name=''):
        self.function = function
        inputs = []
        for arg_name in inspect.signature(function).parameters.keys():
            inputs.append(TypedVariable(dc.type_from_annotation(function.__annotations__[arg_name]), 
                                        arg_name))
        outputs = [TypedVariable(dc.type_from_annotation(function.__annotations__['return']),
                                 'Output function')]

        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return int(hash(self.function.__name__) % 10e5)

    def equivalent(self, other_block):
        return self.method == other_block.method

    def evaluate(self, values):
        return self.function(*values)

class Sequence(Block):

    def __init__(self, number_arguments, type_=None, name=''):
        self.number_arguments = number_arguments
        if type_ is None:
            inputs = [Variable('Sequence element {}'.format(i)) for i in range(self.number_arguments)]
        else:
            inputs = [TypedVariable(type_,
                                    'Sequence element {}'.format(i)) for i in range(self.number_arguments)]

        self.type_ = type_
        outputs = [TypedVariable(list, 'sequence')]
        Block.__init__(self, inputs, outputs, name=name)
        

    def equivalent_hash(self):
        return self.number_arguments

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.number_arguments == other_block.number_arguments


    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['number_arguments'] = self.number_arguments
        if self.type_ is not None:
            dict_['type_'] = dc.serialize_typing(self.type_)
        else:
            dict_['type_'] = None
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        if dict_['type_'] is not None:
            type_ = dc.deserialize_typing(dict_['type_'])
        else:
            type_ = None
        return cls(dict_['number_arguments'], type_, dict_['name'])

    def evaluate(self, values):
        return [[values[var] for var in self.inputs]]


class ForEach(Block):

    def __init__(self, workflow, workflow_iterable_input, name=''):
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
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({
            'workflow': self.workflow.to_dict(),
            'workflow_iterable_input': self.workflow.variable_indices(self.workflow_iterable_input)
            })
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        workflow = Workflow.dict_to_object(dict_['workflow'])
        ib, _, ip = dict_['workflow_iterable_input']

        workflow_iterable_input = workflow.blocks[ib].inputs[ip]
        return cls(workflow, workflow_iterable_input, name=dict_['name'])

    def evaluate(self, values):
        values_workflow = {var2: values[var1] for var1, var2 in zip(self.inputs,
                                                                    self.workflow.inputs)}
        output_values = []
        for value in values_workflow[self.workflow_iterable_input]:
            values_workflow2 = {var.name: val\
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
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'filters': self.filters})
        return dict_    

    @classmethod
    @set_block_variable_names_from_dict
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
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(dict_['attribute_name'], dict_['name'])

    def evaluate(self, values):
        return [getattr(values[self.inputs[0]], self.attribute_name)]


class Sum(Block):

    def __init__(self, number_elements=2, name=''):
        self.number_elements = number_elements
        inputs = []
        inputs = [Variable(name='Sum element {}'.format(i+1)) for i in range(number_elements)]
        outputs = [Variable('Sum')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return self.number_elements

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.number_elements == other_block.number_elements


    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'number_elements': self.number_elements})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(dict_['number_elements'], dict_['name'])


    def evaluate(self, values):
        return sum(values)
    
    
class Substraction(Block):

    def __init__(self, name=''):
        inputs = []
        inputs = [Variable(name='+'),
                  Variable(name='-')]
        outputs = [Variable('Substraction')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return 0

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return True

    def to_dict(self):
        dict_ = Block.to_dict(self)
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        return cls(dict_['name'])


    def evaluate(self, values):
        return [values[self.inputs[0]] - values[self.inputs[1]]]

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

class WorkflowError(Exception):
    pass

class Workflow(Block):
    _standalone_in_db = True
    _allowed_methods = ['run']

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


    def __init__(self, blocks, pipes, output, *, imposed_variable_values=None, name=''):
        self.blocks = blocks
        self.pipes = pipes

        if imposed_variable_values is None:
            self.imposed_variable_values = {}
        else:
            self.imposed_variable_values = imposed_variable_values

        self.coordinates = {}

        self.nonblock_variables = []
        self.variables = []
        for block in self.blocks:
            if isinstance(block, Workflow):
                raise ValueError("Using workflow as blocks is forbidden, use WorkflowBlock wrapper instead")
            self.variables.extend(block.inputs)
            self.variables.extend(block.outputs)
            try:
                self.coordinates[block] = (0, 0)
            except ValueError:
                raise ValueError("can't serialize block {} ({})".format(block, block.name))

        for pipe in self.pipes:
            if not pipe.input_variable in self.variables:
                self.variables.append(pipe.input_variable)
                self.nonblock_variables.append(pipe.input_variable)
            if not pipe.output_variable in self.variables:
                self.variables.append(pipe.output_variable)
                self.nonblock_variables.append(pipe.output_variable)

        self._utd_graph = False

        input_variables = []

        for variable in self.variables:
            if (variable not in self.imposed_variable_values) and\
                (len(nx.ancestors(self.graph, variable)) == 0): # !!! Why not just : if nx.ancestors(self.graph, variable) ?
                # if not hasattr(variable, 'type_'):
                #     raise WorkflowError('Workflow as an untyped input variable: {}'.format(variable.name))
                input_variables.append(variable)

        Block.__init__(self, input_variables, [output], name=name)
        self.output = self.outputs[0]

    def __hash__(self):
        base_hash = len(self.blocks)+11*len(self.pipes)+sum(self.variable_indices(self.outputs[0]))
        block_hash = int(sum([b.equivalent_hash() for b in self.blocks]) % 10e5)
        return base_hash + block_hash

    def __eq__(self, other_workflow):
        # TODO: implement imposed_variable_values in equality
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

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        blocks = [b.__deepcopy__(memo=memo) for b in self.blocks]
        nonblock_variables = [v.__deepcopy(memo=memo) for v in self.nonblock_variables]
        pipes = []
        for pipe in self.pipes:
            input_index = self.variable_indices(pipe.input_variable)
            pipe_input = self.variable_from_index(input_index, blocks, nonblock_variables)

            output_index = self.variable_indices(pipe.output_variable)
            pipe_output = self.variable_from_index(output_index, blocks, nonblock_variables)

            copied_pipe = Pipe(pipe_input, pipe_output)
            memo[pipe] = copied_pipe

            pipes.append(copied_pipe)

        output = self.variable_from_index(self.variable_indices(self.output),
                                          blocks, nonblock_variables)


        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            imposed_variable_values[memo[variable]] = value


        copied_workflow = Workflow(blocks, pipes, output,
                                   imposed_variable_values=imposed_variable_values,
                                   name=self.name)
        return copied_workflow

    def _display_angular(self):
        displays = []
        data = self.jointjs_data()
        displays.extend([{'angular_component': 'workflow',
                         'blocks': data['blocks'],
                         'nonblock_variables': data['nonblock_variables'],
                         'edges': data['edges']}])
        return displays

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        blocks = [b.to_dict() for b in self.blocks]
        # blocks = []
        # for b in self.blocks:
            # blocks.append(b.to_dict())
        pipes = []
        for pipe in self.pipes:
            pipes.append((self.variable_indices(pipe.input_variable),
                          self.variable_indices(pipe.output_variable)))



        dict_.update({'blocks': blocks,
                      'pipes': pipes,
                      'output': self.variable_indices(self.outputs[0]),
                      'nonblock_variables': [v.to_dict() for v in self.nonblock_variables],
                      'package_mix': self.package_mix()})


        imposed_variables = []
        imposed_variable_values = []
        for variable, value in self.imposed_variable_values.items():
            imposed_variables.append(self.variable_indices(variable))
            if hasattr(value, 'to_dict'):
                imposed_variable_values.append(value.to_dict())
            else:
                imposed_variable_values.append(value)

        dict_['imposed_variables'] = imposed_variables
        dict_['imposed_variable_values'] = imposed_variable_values 

        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        blocks = [dc.DessiaObject.dict_to_object(d) for d in dict_['blocks']]
        if 'nonblock_variables' in dict_:
            nonblock_variables = [dc.dict_to_object(d) for d in dict_['nonblock_variables']]
        else:
            nonblock_variables = []

        pipes = []
        for source, target in dict_['pipes']:
            if type(source) == int:
                variable1 = nonblock_variables[source]
            else:
                ib1, _, ip1 = source
                variable1 = blocks[ib1].outputs[ip1]

            if type(target) == int:
                variable2 = nonblock_variables[target]
            else:
                ib2, _, ip2 = target
                variable2 = blocks[ib2].inputs[ip2]

            pipes.append(Pipe(variable1, variable2))

        output = blocks[dict_['output'][0]].outputs[dict_['output'][2]]


        if ('imposed_variable_values' in dict_) and ('imposed_variables' in dict_):
            imposed_variable_values = {}
            for variable_index, serialized_value in zip(dict_['imposed_variables'],
                                                        dict_['imposed_variable_values']):
                # if 'object_class' in serialized_value:
                #     value = dc.DessiaObject.dict_to_object(serialized_value)
                # else:
                value = dc.deserialize(serialized_value)

                if type(variable_index) == int:
                    variable = nonblock_variables[variable_index]
                else:
                    iblock, side, iport = variable_index
                    if side:
                        variable = blocks[iblock].outputs[iport]
                    else:
                        variable = blocks[iblock].inputs[iport]

                imposed_variable_values[variable] = value
        else:
            imposed_variable_values = None

        return cls(blocks=blocks, pipes=pipes, output=output,
                   imposed_variable_values=imposed_variable_values,
                   name=dict_['name'])

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

    @classmethod
    def variable_from_index(cls, index, blocks, nonblock_variables):

        if type(index) == int:
            variable = nonblock_variables[index]
        else:
            ib, side, ip = index
            if not side:
                variable = blocks[ib].inputs[ip]
            else:
                variable = blocks[ib].outputs[ip]
        return variable

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
                return (ib1, ti1, iv1)
            if variable in block.outputs:
                ib1 = iblock
                ti1 = 1
                iv1 = block.outputs.index(variable)
                return (ib1, ti1, iv1)

        # Free variable not attached to block
        if variable in self.nonblock_variables:
            return self.nonblock_variables.index(variable)

        raise WorkflowError('Some thing is wrong with variable {}'.format(variable.name))

    def output_disconnected_elements(self):
        disconnected_elements = []
        ancestors = nx.ancestors(self.graph, self.output)
        for block in self.blocks:
            if not block in ancestors:
                disconnected_elements.append(block)

        for variable in self.nonblock_variables:
            if not variable in ancestors:
                disconnected_elements.append(variable)
        return disconnected_elements


    def index(self, variable):
        index = self.inputs.index(variable)
        return index

    def layout(self, min_horizontal_spacing=300, min_vertical_spacing=200, max_height=800, max_length=1500):
        # block_width = 220
        # block_height = 120
        coordinates = {}
        elements_by_distance = {}
        for element in self.blocks+self.nonblock_variables:
            distances = []
            paths = nx.all_simple_paths(self.graph, element, self.outputs[0])
            for path in paths:
                distance = 0
                for path_element in path[1:-1]:
                    if path_element in self.blocks:
                        distance += 1
                    elif path_element in self.nonblock_variables:
                        distance += 1
                distances.append(distance)
            try:
                distance = max(distances)
            except ValueError:
                distance = 3
            if distance in elements_by_distance:
                elements_by_distance[distance].append(element)
            else:
                elements_by_distance[distance] = [element]

        if len(elements_by_distance) != 0:
            max_distance = max(elements_by_distance.keys())
        else:
            max_distance = 3 # TODO: this is an awfull quick fix

        horizontal_spacing = max(min_horizontal_spacing, max_length/max_distance)

        for i, distance in enumerate(sorted(elements_by_distance.keys())[::-1]):
            n = len(elements_by_distance[distance])
            # if n == 0:
            #     n = 1
            vertical_spacing = min(min_vertical_spacing, max_height/n)
            horizontal_anchor_size = max_distance
            for j, element in enumerate(elements_by_distance[distance]):
                coordinates[element] = (i*horizontal_spacing, (j+0.5)*vertical_spacing)

        return coordinates

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
        nx.draw_networkx_labels(self.graph, pos, labels)


    def run(self, input_variables_values, verbose=False,
            progress_callback=None):
        log = ''
        activated_items = {p: False for p in self.pipes}
        activated_items.update({v: False for v in self.variables})
        activated_items.update({b: False for b in self.blocks})

        values = {}
        # Imposed variables values activation
        for variable, value in self.imposed_variable_values.items():
            # Type checking
            value_type_check(value, variable.type_)
                # raise ValueError('Bad type', value, variable.type_)
            values[variable] = value
            activated_items[variable] = True
            
        # Input activation
        for index, variable in enumerate(self.inputs):
            if index in input_variables_values:
                value = input_variables_values[index]
                typeguard.check_type(variable.name, value, variable.type_)
                    # raise ValueError('Bad type', value, variable.type_)

                values[variable] = value
                activated_items[variable] = True
            elif hasattr(variable, 'default_value'):
                values[variable] = variable.default_value
                activated_items[variable] = True
            else:
                raise ValueError('Value {} has no value'.format(variable.name))

        something_activated = True

        start_time = time.time()

        log_line = 'Starting workflow run at {}'.format(time.strftime('%d/%m/%Y %H:%M:%S UTC',
                                                                      time.gmtime(start_time)))
        log += (log_line + '\n')
        if verbose:
            print(log_line)
        progress = 0

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
                            log_line = 'Evaluating block {}'.format(block.name)
                            log += log_line + '\n'
                            if verbose:
                                print(log_line)
                        output_values = block.evaluate({i: values[i]\
                                                        for i in block.inputs})
                        # Updating progress
                        if progress_callback is not None:
                            progress += 1/len(self.blocks)
                            progress_callback(progress)
                            
                        # Unpacking result of evaluation
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

    def interactive_input_variables_values(self):
        input_variables_values = {}
        for i, input_ in enumerate(self.inputs):
            print('Input n°{}: {} type: {}: '.format(i+1, input_.name, input_.type_))
            if hasattr(input_, 'default_value'):                
                value = input('Value? default={} '.format(input_.default_value))
                if value=='':
                    value = input_.default_value
            else:
                value = input_.type_(input('Value? '))
                
            input_variables_values[i] = value
            
        for i in sorted(input_variables_values.keys()):
            input_ = self.inputs[i]
            value = input_variables_values[i]
            print('{}/ {}: {}'.format(i, value))
            
        return input_variables_values

    def mxgraph_data(self):
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
            edges.append((self.variable_indices(pipe.input_variable),
                          self.variable_indices(pipe.output_variable)))

        return nodes, edges


    def jointjs_data(self):
        coordinates = self.layout()
        blocks = []
        for block in self.blocks:
            # !!! Is it necessary to add is_workflow_input/output for outputs/inputs ??
            block_data = block.jointjs_data()
            block_data.update({'inputs': [{'name': i.name,
                                           'is_workflow_input': i in self.inputs,
                                           'has_default_value': hasattr(i, 'default_value')}\
                                          for i in block.inputs],

                           'outputs': [{'name': o.name,
                                        'is_workflow_output': o in self.outputs}\
                                       for o in block.outputs],
                           'position': coordinates[block]})
            blocks.append(block_data)

        nonblock_variables = []
        for variable in self.nonblock_variables:
            nonblock_variables.append({'name': variable.name,
                                       'is_workflow_input': variable in self.inputs,
                                       'position': coordinates[variable]})

        edges = []
        for pipe in self.pipes:
            input_index = self.variable_indices(pipe.input_variable)
            if type(input_index) == int:
                node1 = input_index
            else:
                ib1, is1, ip1 = input_index
                if is1:
                    block = self.blocks[ib1]
                    ip1 += len(block.inputs)

                node1 = [ib1, ip1]

            output_index = self.variable_indices(pipe.output_variable)
            if type(output_index) == int:
                node2 = output_index
            else:
                ib2, is2, ip2 = output_index
                if is2:
                    block = self.blocks[ib2]
                    ip2 += len(block.inputs)

                node2 = [ib2, ip2]

            edges.append([node1, node2])

        data = Block.jointjs_data(self)
        data.update({'blocks': blocks,
                    'nonblock_variables': nonblock_variables,
                    'edges': edges})
        return data


    # def plot_mxgraph(self):
    #     env = Environment(loader=PackageLoader('dessia_common', 'templates'),
    #                       autoescape=select_autoescape(['html', 'xml']))


    #     template = env.get_template('workflow.html')

    #     mx_path = pkg_resources.resource_filename(pkg_resources.Requirement('dessia_common'),
    #                                               'dessia_common/templates/mxgraph')

    #     nodes, edges = self.mxgraph_data()
    #     options = {}
    #     rendered_template = template.render(mx_path=mx_path,
    #                                         nodes=nodes,
    #                                         edges=edges,
    #                                         options=options)

    #     temp_file = tempfile.mkstemp(suffix='.html')[1]

    #     with open(temp_file, 'wb') as file:
    #         file.write(rendered_template.encode('utf-8'))

    #     webbrowser.open('file://' + temp_file)


    def plot_jointjs(self):
        env = Environment(loader=PackageLoader('dessia_common', 'templates'),
                          autoescape=select_autoescape(['html', 'xml']))


        template = env.get_template('workflow_jointjs.html')

        data = self.jointjs_data()
        options = {}
        rendered_template = template.render(blocks=json.dumps(data['blocks']),
                                            nonblock_variables=json.dumps(data['nonblock_variables']),
                                            edges=data['edges'],
                                            options=options)

        temp_file = tempfile.mkstemp(suffix='.html')[1]

        with open(temp_file, 'wb') as file:
            file.write(rendered_template.encode('utf-8'))

        webbrowser.open('file://' + temp_file)


    
    def is_valid(self):
        # Checking types of each end of pipes
        for pipe in self.pipes:
            if hasattr(pipe.input_variable, 'type_') and hasattr(pipe.output_variable, 'type_'):
                type1 = pipe.input_variable.type_
                type2 = pipe.output_variable.type_
                if type1 != type2:
                    # print(type1, type2)
                    try:
                        consistent = issubclass(pipe.input_variable.type_, pipe.output_variable.type_)
                        
                    except TypeError:
                        # TODO: need of a real typing check
                        consistent = True
        
                        if not consistent:
                            raise TypeError('inconsistent pipe type from pipe input {} to pipe output {}: {} incompatible with {}'.format(
                                pipe.input_variable.name, pipe.output_variable.name,
                                pipe.input_variable.type_, pipe.output_variable.type_)
                                )
        return True

    def package_mix(self):
        """
        
        """
        package_mix = {}
        for block in self.blocks:
            if hasattr(block, 'package_mix'):
                for package_name, fraction in block.package_mix().items():
                    if package_name in package_mix:
                        package_mix[package_name] += fraction
                    else:
                        package_mix[package_name] = fraction

        # Adimension
        fraction_sum = sum(package_mix.values())
        return {pn:f/fraction_sum for pn, f in package_mix.items()}

class WorkflowBlock(Block):
    """
    Wrapper around workflow to put it in a block of another workflow
    Even if a workflow is a block, it can't be used directly as it has a different behavior 
    than a Block in eq and hash which is problematic to handle in dicts for example
    """

    def __init__(self, workflow:Workflow, name:str=''):
        self.workflow = workflow

        inputs = []
        for variable in self.workflow.inputs:
            input_ = variable.copy()
            input_.name = '{} - {}'.format(name, variable.name)
            inputs.append(input_)

        outputs = [self.workflow.output.copy()]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return hash(self.workflow)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.workflow == other_block.workflow

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'workflow': self.workflow.to_dict()})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        b = cls(Workflow.dict_to_object(dict_['workflow']),
                   name=dict_['name'])

        return b

    def evaluate(self, values):
        args = {self.inputs.index(input_): v for input_, v in values.items()}
        workflow_run = self.workflow.run(args)
        return [workflow_run.output_value]
    
    def package_mix(self):
        return self.workflow.package_mix()


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
                values = [(i, {f['attribute'] : dc.getdeepattr(v, f['attribute'])
                               for f in filter_display[0]['filters']})\
                          for i, v in enumerate(self.output_value)]
                filter_display[0]['datasets'] = [{'label' : 'Results',
                                                  'values' : values,
                                                  'color' : "#99b4d6"}]
                filter_display[0]['references_attribute'] = 'output_value'
                displays.extend(filter_display)
        return displays

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'workflow' : self.workflow.to_dict(),
                      'start_time' : self.start_time,
                      'end_time' : self.end_time,
                      'execution_time' : self.execution_time,
                      'log' : self.log})

        if self.output_value is not None:
            dict_.update({'output_value' : dc.serialize_sequence(self.output_value),
                          'output_value_type' : dc.recursive_type(self.output_value)})

        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        workflow = Workflow.dict_to_object(dict_['workflow'])
        if 'output_value' in dict_ and 'output_value_type' in dict_:
            output_value = dc.recursive_instantiation(dict_['output_value_type'], dict_['output_value'])
        else:
            output_value = None
        return cls(workflow=workflow, output_value=output_value,
                   start_time=dict_['start_time'], end_time=dict_['end_time'],
                   log=dict_['log'], name=dict_['name'])

def set_inputs_from_function(method, inputs=[], block_name=''):
    """
    
    """
    args_specs = inspect.getfullargspec(method)
    nargs = len(args_specs.args) - 1

    if args_specs.defaults is not None:
        ndefault_args = len(args_specs.defaults)
    else:
        ndefault_args = 0

    for iargument, argument in enumerate(args_specs.args[1:]):
        if not argument in ['self', 'progress_callback']:
            type_ = dc.type_from_annotation(method.__annotations__[argument], module=method.__module__)
            if iargument >= nargs - ndefault_args:
                default_value = args_specs.defaults[ndefault_args-nargs+iargument]
                inputs.append(TypedVariableWithDefaultValue(type_,
                                                            default_value=default_value,
                                                            name=argument))
                
            else:
                inputs.append(TypedVariable(type_=type_,
                                            name=argument))

    return inputs

def value_type_check(value, type_):
    try:
        typeguard.check_type('', value, type_)
    except TypeError:
        return False
    try:# TODO: Subscripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass
            
    return True
    