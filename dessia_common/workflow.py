#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import inspect
import time
import tempfile
import json
from importlib import import_module
import webbrowser
import networkx as nx
from typing import List
from copy import deepcopy
from dessia_common.templates import workflow_template

import dessia_common as dc
from dessia_common.vectored_objects import ParetoSettings, from_csv
import plot_data
from plot_data.colors import BLUE, LIGHTBLUE, LIGHTGREY


class Variable(dc.DessiaObject):
    _standalone_in_db = False
    _eq_is_data_eq = False

    def __init__(self, memorize: bool = False, name: str = ''):
        self.memorize = memorize
        dc.DessiaObject.__init__(self, name=name)


class TypedVariable(Variable):
    def __init__(self, type_, memorize: bool = False, name: str = ''):
        Variable.__init__(self, memorize=memorize, name=name)
        self.type_ = type_

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'type': dc.serialize_typing(self.type_),
                      'memorize': self.memorize})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        type_ = dc.get_python_class_from_class_name(dict_['type'])
        memorize = dict_['memorize']
        return cls(type_=type_, memorize=memorize, name=dict_['name'])


class VariableWithDefaultValue(Variable):
    def __init__(self, default_value, memorize: bool = False, name: str = ''):
        Variable.__init__(self, memorize=memorize, name=name)
        self.default_value = default_value


class TypedVariableWithDefaultValue(TypedVariable):
    def __init__(self, type_, default_value, memorize: bool = False, name=''):
        TypedVariable.__init__(self, type_=type_, memorize=memorize, name=name)
        self.default_value = default_value

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'type': dc.serialize_typing(self.type_),
                      'default_value': dc.serialize(self.default_value),
                      'memorize': self.memorize})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        type_ = dc.get_python_class_from_class_name(dict_['type_'])
        default_value = dc.deserialize(dict_['default_value'])
        return cls(type_=type_, default_value=default_value,
                   memorize=dict_['memorize'], name=dict_['name'])

    def copy(self):
        return TypedVariableWithDefaultValue(type_=self.type_,
                                             default_value=self.default_value,
                                             memorize=self.memorize,
                                             name=self.name)


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
    _eq_is_data_eq = False
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
    
    def jointjs_data(self):
        data = {'block_class': self.__class__.__name__}
        if self.name != '':
            data['name'] = self.name
        else:
            data['name'] = self.__class__.__name__
        return data


class Import(Block):
    def __init__(self, type_: str, name: str = ''):
        self.type_ = type_
        inputs = [TypedVariable(type_=str, name='Input filename'),
                  TypedVariableWithDefaultValue(type_=bool, default_value=True,
                                                name='Remove duplicates')]
        outputs = [Variable(name='Array'), Variable(name='Variables')]

        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

    def equivalent_hash(self):
        return len(self.type_)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.type_ == other_block.type_

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['type_'] = self.type_
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(type_=dict_['type_'], name=dict_['name'])

    def evaluate(self, values):
        dirname = os.path.dirname(__file__)
        relative_filepath = 'models/data/'+values[self.inputs[0]]
        filename = os.path.join(dirname, relative_filepath)
        if self.type_ == 'csv':
            array, variables = from_csv(filename=filename, end=None,
                                        remove_duplicates=True)
            return [array, variables]
        msg = 'File type {} not supported'.format(self.type_)
        raise NotImplementedError(msg)


class InstanciateModel(Block):
    """
    :param model_class: The class to instanciate.
    :type model_class: DessiaObject
    :param name: The name of the block.
    :type name: str
    """
    _jsonschema = dc.dict_merge(Block._jsonschema, {
        "title": "Instantiate model Base Schema",
        "required": ['object_class'],
        "properties": {
            "object_class": {
                "type": "string",
                "editable": True,
                "examples": ['Nom']
                }
            }
        })

    def __init__(self, model_class, name=''):
        self.model_class = model_class
        inputs = []

        inputs = set_inputs_from_function(self.model_class.__init__, inputs, name)

        outputs = [TypedVariable(type_=self.model_class,
                                 name='Instanciated object')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.model_class.__name__)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        classname = self.model_class.__class__.__name__
        other_classname = other_block.model_class.__class__.__name__
        return classname == other_classname

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'model_class': self.model_class.__name__,
                      'model_class_module': self.model_class.__module__})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
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

        type_ = dc.type_from_annotation(method.__annotations__['return'],
                                        method.__module__)
        output_name = 'method result of {}'.format(self.method_name)
        outputs = [TypedVariable(type_=type_, name=output_name)]
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
        class_ = getattr(import_module(dict_['class_module']),
                         dict_['class_'])
        return cls(class_=class_,
                   method_name=dict_['method_name'],
                   name=dict_['name'])

    def evaluate(self, values):
        args = {arg_name: values[var]
                for arg_name, var in zip(self.argument_names, self.inputs)
                if var in values}
        return [getattr(self.class_, self.method_name)(**args)]


class ModelMethod(Block):
    """
    :param model_class: The class owning the method.
    :type model_class: DessiaObject
    :param method_name: The name of the method.
    :type method_name: str
    :param name: The name of the block.
    :type name: str
    """
    _jsonschema = dc.dict_merge(Block._jsonschema, {
        "title": "Model method Base Schema",
        "required": ['model_class', 'method_name'],
        "properties": {
            "model_class": {
                "type": "string",
                "editable": True,
                "examples": ['Nom']
                },

            "method_name": {
                "type": "string",
                "editable": True,
                "examples": ['Nom']
                }
            }
        })

    def __init__(self, model_class, method_name, name=''):
        self.model_class = model_class
        self.method_name = method_name
        inputs = [TypedVariable(type_=model_class, name='model at input')]
        method = getattr(self.model_class, self.method_name)

        inputs = set_inputs_from_function(method, inputs, name)

        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        result_output_name = 'method result of {}'.format(self.method_name)
        if 'return' in method.__annotations__:
            type_ = dc.type_from_annotation(method.__annotations__['return'],
                                            method.__module__)
            return_output = TypedVariable(type_=type_, name=result_output_name)
        else:
            return_output = Variable(name=result_output_name)

        model_output_name = 'model at output {}'.format(self.method_name)
        model_output = TypedVariable(type_=model_class, name=model_output_name)
        outputs = [return_output, model_output]
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
        class_ = getattr(import_module(dict_['model_class_module']),
                         dict_['model_class'])
        return cls(model_class=class_,
                   method_name=dict_['method_name'],
                   name=dict_['name'])

    def evaluate(self, values):
        args = {arg_name: values[var]
                for arg_name, var in zip(self.argument_names, self.inputs[1:])
                if var in values}
        return [getattr(values[self.inputs[0]], self.method_name)(**args),
                values[self.inputs[0]]]

    def package_mix(self):
        return {self.model_class.__module__.split('.')[0]: 1}


class Function(Block):
    def __init__(self, function, name=''):
        self.function = function
        inputs = []
        for arg_name in inspect.signature(function).parameters.keys():
            # TODO: Check why we need TypedVariables
            type_ = dc.type_from_annotation(function.__annotations__[arg_name])
            inputs.append(TypedVariable(type_=type_, name=arg_name))
        out_type = dc.type_from_annotation(function.__annotations__['return'])
        outputs = [TypedVariable(type_=out_type, name='Output function')]

        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return int(hash(self.function.__name__) % 10e5)

    def equivalent(self, other_block):
        # TODO : chenge method to function
        return self.method == other_block.method

    def evaluate(self, values):
        return self.function(*values)


class Sequence(Block):
    def __init__(self, number_arguments, type_=None, name=''):
        self.number_arguments = number_arguments
        prefix = 'Sequence element {}'
        if type_ is None:
            inputs = [Variable(name=prefix.format(i))
                      for i in range(self.number_arguments)]
        else:
            inputs = [TypedVariable(type_=type_, name=prefix.format(i))
                      for i in range(self.number_arguments)]

        self.type_ = type_
        outputs = [TypedVariable(type_=list, name='sequence')]
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
    """
    :param workflow_block: The WorkflowBlock on which iterate.
    :type workflow_block: WorkflowBlock
    :param workflow_iterable_input: The iterable input of the workflow.
    :type workflow_iterable_input: TypedVarible
    :param name: The name of the block.
    :type name: str
    """
    def __init__(self, workflow_block, workflow_iterable_input, name=''):
        self.workflow_block = workflow_block
        self.workflow_iterable_input = workflow_iterable_input
        inputs = []

        for workflow_input in self.workflow_block.inputs:
            if workflow_input == workflow_iterable_input:
                name = 'Iterable input: ' + workflow_input.name
                inputs.append(Variable(name=name))
            else:
                input_ = workflow_input.copy()
                input_.name = 'binding '+input_.name
                inputs.append(input_)
        output_variable = Variable(name='Foreach output')

        Block.__init__(self, inputs, [output_variable], name=name)

    def equivalent_hash(self):
        return int(self.workflow_block.equivalent_hash() % 10e5)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False

        # TODO Check for non-resolved variable_indices in workflow_bloc
        return self.workflow_block == other_block.workflow_block\
        and self.workflow_block.variable_indices(self.workflow_iterable_input)\
        == other_block.workflow_block.variable_indices(other_block.workflow_iterable_input)

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({
            'workflow_block': self.workflow_block.to_dict(),
            'workflow_iterable_input': self.workflow_block.inputs.index(self.workflow_iterable_input)
            })
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        workflow_block = WorkflowBlock.dict_to_object(dict_['workflow_block'])
        index = dict_['workflow_iterable_input']

        workflow_iterable_input = workflow_block.inputs[index]
        return cls(workflow_block, workflow_iterable_input, name=dict_['name'])

    def evaluate(self, values):

        values_workflow = {var2: values[var1]
                           for var1, var2 in zip(self.inputs,
                                                 self.workflow_block.inputs)}
        output_values = []
        for value in values_workflow[self.workflow_iterable_input]:
            values_workflow[self.workflow_iterable_input] = value
            output_values.append(self.workflow_block.evaluate(values_workflow)[0])
        return [output_values]


class Unpacker(Block):
    def __init__(self, indices: List[int], name: str = ''):
        self.indices = indices
        inputs = [Variable(name='input_sequence')]
        outputs = [Variable(name='output_{}'.format(i)) for i in indices]

        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.indices == other_block.indices

    def equivalent_hash(self):
        return len(self.indices)

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['indices'] = self.indices
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        return cls(dict_['indices'], dict_['name'])

    def evaluate(self, values):
        return [values[self.inputs[0]][i] for i in self.indices]


class Flatten(Block):
    def __init__(self, name=''):
        inputs = [Variable(name='input_sequence')]
        outputs = [Variable(name='flatten_sequence')]
        Block.__init__(self, inputs, outputs, name=name)
        
    def equivalent_hash(self):
        return 1
    
    @classmethod
    def dict_to_object(cls, dict_):
        return cls(dict_['name'])
    
    def evaluate(self, values):
        output = []
        for value in values[self.inputs[0]]:
            output.extend(value)
        return [output]


class Filter(Block):
    """ 
    :param filters: A list of dictionaries, each corresponding to a value to filter. \
    The dictionary should be as follows : *{'attribute' : the name of the attribute to \
    filter (str), 'operator' : choose between gt, lt, get, let (standing for greater \
    than, lower than, geater or equal than, lower or equal than) (str), 'bound' : \
    the value (float)}*
    :type filters: list[dict]
    :param name: The name of the block.
    :type name: str
    """
    def __init__(self, filters, name=''):
        self.filters = filters
        inputs = [Variable(name='input_list')]
        outputs = [Variable(name='output_list')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.filters == other_block.filters

    def equivalent_hash(self):
        return int(sum([hash(v) for f in self.filters for v in f.values()]) % 10e5)

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
            bounded = True
            i = 0
            while bounded and i < len(self.filters):
                filter_ = self.filters[i]
                value = dc.enhanced_deep_attr(object_, filter_['attribute'])
                bounded = dc.is_bounded(filter_, value)
                i += 1

            if bounded:
                ouput_values.append(object_)
        return [ouput_values]


class ParallelPlot(Block):
    """
    :param attributes: A List of all attributes that will be shown inside the \
    ParallelPlot window on the DessIA Platform.
    :type attributes: List[str]
    :param name: The name of the block.
    :type name: str
    """
    def __init__(self, attributes: List[str], order: int = 0, name: str = ''):
        self.attributes = attributes
        self.order = order
        pareto_input = TypedVariableWithDefaultValue(type_=ParetoSettings,
                                                     default_value=None,
                                                     memorize=True,
                                                     name='Pareto settings')
        inputs = [Variable(memorize=True, name='input_list'), pareto_input]
        outputs = []
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        equal = self.attributes == other_block.attributes\
            and self.order == other_block.order
        return equal

    def equivalent_hash(self):
        return sum([len(a) for a in self.attributes]) + self.order

    def _display(self, local_values):
        objects = local_values[self.inputs[0]]
        # pareto_settings = local_values[self.inputs[1]]

        values = [{a: dc.enhanced_deep_attr(o, a) for a in self.attributes}
                  for o in objects]
        # for object_ in objects:
        #     value = {}
        #     for attribute in self.attributes:
        #         value[attribute] = dc.enhanced_deep_attr(object_, attribute)
        #     values.append(value)

        first_vars = self.attributes[:2]
        values2d = [{key: val[key]} for key in first_vars for val in
                    values]
        rgbs = [[192, 11, 11], [14, 192, 11], [11, 11, 192]]

        tooltip = plot_data.Tooltip(name='Tooltip',
                                    to_plot_list=self.attributes)

        scatterplot = plot_data.Scatter(axis=plot_data.DEFAULT_AXIS,
                                        tooltip=tooltip,
                                        to_display_att_names=first_vars,
                                        point_shape='circle', point_size=2,
                                        color_fill=LIGHTGREY,
                                        color_stroke=BLUE,
                                        stroke_width=1, elements=values2d,
                                        name='Scatter Plot')

        parallelplot = plot_data.ParallelPlot(line_color=LIGHTBLUE,
                                              line_width=1,
                                              disposition='horizontal',
                                              to_disp_attributes=self.attributes,
                                              rgbs=rgbs, elements=values)
        objects = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300),
                 plot_data.Window(width=560, height=300)]
        coords = [(0, 0), (0, 300)]
        multiplot = plot_data.MultiplePlots(points=values, objects=objects,
                                            sizes=sizes, coords=coords,
                                            name='Results plot')
        dict_ = multiplot.to_dict()
        dict_['references_attribute'] = 'output_value'
        displays = [{'angular_component': 'plot_data', 'data': dict_}]
        return displays

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'attributes': self.attributes, 'order': self.order})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(attributes=dict_['attributes'],
                   order=dict_['order'],
                   name=dict_['name'])

    @staticmethod
    def evaluate(self):
        return []


class Display(Block):
    def __init__(self, order: int = 0, name: str = ''):
        self.order = order
        inputs = [Variable(name='Model to Display', memorize=True)]
        outputs = []

        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

    def equivalent(self, other_block):
        if not Block.equivalent(self, other_block):
            return False
        return self.order == other_block.order

    def equivalent_hash(self):
        return self.order

    def _display(self, local_values):
        object_ = local_values[self.inputs[0]]
        displays = object_._display_angular()
        return displays

    def to_dict(self):
        dict_ = dc.DessiaObject.base_dict(self)
        dict_['order'] = self.order
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(order=dict_['order'], name=dict_['name'])

    @staticmethod
    def evaluate(self):
        return []


class ModelAttribute(Block):
    """
    :param attribute_name: The name of the attribute to select.
    :type attribute_name: str
    :param name: The name of the block.
    :type name: str
    """
    def __init__(self, attribute_name, name=''):
        self.attribute_name = attribute_name

        inputs = [Variable(name='Model')]
        outputs = [Variable(name='Model attribute')]
        Block.__init__(self, inputs, outputs, name=name)

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
        return [dc.enhanced_deep_attr(values[self.inputs[0]],
                                      self.attribute_name)]


class Sum(Block):
    def __init__(self, number_elements=2, name=''):
        self.number_elements = number_elements
        inputs = [Variable(name='Sum element {}'.format(i+1))
                  for i in range(number_elements)]
        outputs = [Variable(name='Sum')]
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
        return [sum(values)]
    
    
class Substraction(Block):
    def __init__(self, name=''):
        inputs = [Variable(name='+'), Variable(name='-')]
        outputs = [Variable(name='Substraction')]
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
    """
    :param input_variable: The input varaible of the pipe correspond to the \
    start of the arrow, its tail.
    :type input_variable: TypedVariable
    :param output_variable: The output variable of the pipe correpond to the \
    end of the arrow, its hat.
    :type output_variable: TypedVariable
    """
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
                "classes": [
                    "dessia_common.workflow.Variable",
                    "dessia_common.workflow.TypedVariable",
                    "dessia_common.workflow.VariableWithDefaultValue",
                    "dessia_common.workflow.TypedVariableWithDefaultValue"
                ]
            },
            "output_variable": {
                "type": "object",
                "editable": True,
                "classes": [
                    "dessia_common.workflow.Variable",
                    "dessia_common.workflow.TypedVariable",
                    "dessia_common.workflow.VariableWithDefaultValue",
                    "dessia_common.workflow.TypedVariableWithDefaultValue"
                ],
            }
        }
    }
    _eq_is_data_eq = False

    def __init__(self, input_variable, output_variable):
        self.input_variable = input_variable
        self.output_variable = output_variable

    def to_dict(self):
        return {'input_variable': self.input_variable,
                'output_variable': self.output_variable}


class WorkflowError(Exception):
    pass


class Workflow(Block):
    """
    :param blocks: A List with all the Blocks used by the Worklow.
    :type blocks: List of Block objects
    :param pipes: A List of Pipe objects.
    :type pipes: List of Pipe objects
    :param imposed_variable_values: A dictionary of imposed variable values.
    :type imposed_variable_values: dict
    :param name: The name of the block.
    :type name: str
    """ 
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
                "editable": True,
                "items": {
                    "type": "object",
                    "classes": ["dessia_common.workflow.InstanciateModel",
                                "dessia_common.workflow.ModelMethod",
                                "dessia_common.workflow.ForEach",
                                "dessia_common.workflow.ModelAttribute",
                                "dessia_common.workflow.Function",
                                "dessia_common.workflow.Sequence",
                                "dessia_common.workflow.ForEach",
                                "dessia_common.workflow.Unpacker",
                                "dessia_common.workflow.Flatten",
                                "dessia_common.workflow.Filter",
                                "dessia_common.workflow.ParallelPlot",
                                "dessia_common.workflow.Sum",
                                "dessia_common.workflow.Substraction"],
                    "editable": True,
                    },
                },
            "pipes": {
                "type": "array",
                "order": 1,
                "editable": True,
                "items": {
                    'type': 'objects',
                    'classes': ["dessia_common.workflow.Pipe"],
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

    def __init__(self, blocks, pipes, output, *,
                 imposed_variable_values=None, name=''):
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
                msg = "Using workflow as blocks is forbidden, "
                msg += "use WorkflowBlock wrapper instead"
                raise ValueError(msg)
            self.variables.extend(block.inputs)
            self.variables.extend(block.outputs)
            try:
                self.coordinates[block] = (0, 0)
            except ValueError:
                msg = "can't serialize block {} ({})".format(block, block.name)
                raise ValueError(msg)

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

    def _data_hash(self):
        base_hash = len(self.blocks)\
                    + 11 * len(self.pipes)\
                    + sum(self.variable_indices(self.outputs[0]))
        block_hash = int(sum([b.equivalent_hash() for b in self.blocks])
                         % 10e5)
        return base_hash + block_hash

    def _data_eq(self, other_workflow):
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
        nonblock_variables = [v.__deepcopy__(memo=memo)
                              for v in self.nonblock_variables]
        pipes = []
        for pipe in self.pipes:
            input_index = self.variable_indices(pipe.input_variable)
            pipe_input = self.variable_from_index(input_index, blocks,
                                                  nonblock_variables)

            output_index = self.variable_indices(pipe.output_variable)
            pipe_output = self.variable_from_index(output_index, blocks,
                                                   nonblock_variables)

            copied_pipe = Pipe(pipe_input, pipe_output)
            memo[pipe] = copied_pipe

            pipes.append(copied_pipe)

        output = self.variable_from_index(self.variable_indices(self.output),
                                          blocks, nonblock_variables)

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            imposed_variable_values[memo[variable]] = value

        copied_workflow = Workflow(blocks=blocks, pipes=pipes, output=output,
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

    @property
    def _method_jsonschemas(self):
        jsonschemas = {'run': deepcopy(dc.JSONSCHEMA_HEADER)}
        properties_dict = jsonschemas['run']['properties']
        required_inputs = []
        for i, input_ in enumerate(self.inputs):
            current_dict = {}
            annotation = (str(i), input_.type_)
            if input_ in self.nonblock_variables:
                title = input_.name
            else:
                input_block = self.block_from_variable(input_)
                if input_block.name:
                    title = dc.prettyname(input_block.name + ' - ' + input_.name)
                else:
                    title = dc.prettyname(input_.name)

            annotation_jsonschema = dc.jsonschema_from_annotation(
                annotation=annotation,
                jsonschema_element=current_dict,
                order=i,
                title=title
            )
            current_dict.update(annotation_jsonschema[str(i)])
            if not isinstance(input_, (VariableWithDefaultValue,
                                       TypedVariableWithDefaultValue)):
                required_inputs.append(str(i))
            else:
                current_dict.update(dc.set_default_value(current_dict,
                                                         str(i),
                                                         input_.default_value))
            properties_dict[str(i)] = current_dict[str(i)]
        jsonschemas['run']['required'] = required_inputs
        return jsonschemas

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
                      'nonblock_variables': [v.to_dict()
                                             for v in self.nonblock_variables],
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
            nonblock_variables = [dc.dict_to_object(d)
                                  for d in dict_['nonblock_variables']]
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

    def dict_to_arguments(self, dict_):
        arguments_values = {}
        for i, input_ in enumerate(self.inputs):
            if not isinstance(input_, (VariableWithDefaultValue, TypedVariableWithDefaultValue))\
            or (isinstance(input_, (VariableWithDefaultValue, TypedVariableWithDefaultValue))
                and str(i) in dict_):
                value = dict_[str(i)]
                deserialized_value = dc.deserialize_argument(input_.type_, value)
                arguments_values[i] = deserialized_value

        arguments = {'input_values': arguments_values}
        return arguments

    @classmethod
    def variable_from_index(cls, index, blocks, nonblock_variables):
        if type(index) == int:
            variable = nonblock_variables[index]
        else:
            # ib, side, ip = index
            if not index[1]:
                variable = blocks[index[0]].inputs[index[2]]
            else:
                variable = blocks[index[0]].outputs[index[2]]
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

        msg = 'Some thing is wrong with variable {}'.format(variable.name)
        raise WorkflowError(msg)

    def block_from_variable(self, variable):
        iblock, _, _ = self.variable_indices(variable)
        return self.blocks[iblock]

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

    def layout(self, min_horizontal_spacing=300,
               min_vertical_spacing=200,
               max_height=800, max_length=1500):
        # block_width = 220
        # block_height = 120
        coordinates = {}
        elements_by_distance = {}
        for element in self.blocks+self.nonblock_variables:
            distances = []
            paths = nx.all_simple_paths(self.graph, element, self.outputs[0])
            for path in paths:
                distance = 1
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

        labels = {}  # b: b.function.__name__ for b in self.block}
        for block in self.blocks:
            labels[block] = block.__class__.__name__
            for variable in self.variables:
                labels[variable] = variable.name
        nx.draw_networkx_labels(self.graph, pos, labels)

    def run(self, input_values, verbose=False,
            progress_callback=None, name=None):
        log = ''
        activated_items = {p: False for p in self.pipes}
        activated_items.update({v: False for v in self.variables})
        activated_items.update({b: False for b in self.blocks})

        values = {}
        variables_values = {}
        # Imposed variables values activation
        for variable, value in self.imposed_variable_values.items():
            # Type checking
            value_type_check(value, variable.type_)
                # raise ValueError('Bad type', value, variable.type_)
            values[variable] = value
            activated_items[variable] = True
            
        # Input activation
        for index, variable in enumerate(self.inputs):
            if index in input_values:
                value = input_values[index]
                # typeguard.check_type(variable.name, value, variable.type_)
                    # raise ValueError('Bad type', value, variable.type_)

                values[variable] = value
                activated_items[variable] = True
            elif hasattr(variable, 'default_value'):
                values[variable] = variable.default_value
                activated_items[variable] = True
            else:
                raise ValueError('Value {} of index {} in inputs has no value'.format(variable.name, index))

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
                        output_values = block.evaluate({i: values[i]
                                                        for i in block.inputs})
                        for input_ in block.inputs:
                            if input_.memorize:
                                indices = str(self.variable_indices(input_))
                                variables_values[indices] = values[input_]
                        # Updating progress
                        if progress_callback is not None:
                            progress += 1/len(self.blocks)
                            progress_callback(progress)
                            
                        # Unpacking result of evaluation
                        for output, output_value in zip(block.outputs, output_values):
                            if output.memorize:
                                indices = str(self.variable_indices(output))
                                variables_values[indices] = output_value
                            values[output] = output_value
                            activated_items[output] = True

                        activated_items[block] = True
                        something_activated = True

        end_time = time.time()
        log_line = 'Workflow terminated in {} s'.format(end_time - start_time)

        log += log_line + '\n'
        if verbose:
            print(log_line)

        output_value = values[self.outputs[0]]
        if not name:
            name = self.name+' run'
        return WorkflowRun(workflow=self, input_values=input_values,
                           output_value=output_value,
                           variables_values=variables_values,
                           start_time=start_time, end_time=end_time,
                           log=log, name=name)

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
            # TOCHECK Is it necessary to add is_workflow_input/output for outputs/inputs ??
            block_data = block.jointjs_data()
            block_data.update({'inputs': [{'name': i.name,
                                           'is_workflow_input': i in self.inputs,
                                           'has_default_value': hasattr(i, 'default_value')}\
                                          for i in block.inputs],
                               'outputs': [{'name': o.name,
                                            'is_workflow_output': o in self.outputs}
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

    def plot(self):
        self.plot_jointjs()


    def plot_jointjs(self):

        data = self.jointjs_data()
        rendered_template = workflow_template.substitute(workflow_data=json.dumps(data))

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
        return {pn: f/fraction_sum for pn, f in package_mix.items()}


class WorkflowBlock(Block):
    """
    Wrapper around workflow to put it in a block of another workflow
    Even if a workflow is a block, it can't be used directly as it has
    a different behavior
    than a Block in eq and hash which is problematic to handle in dicts
    for example
    """

    def __init__(self, workflow: Workflow, name: str = ''):
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
        b = cls(workflow=Workflow.dict_to_object(dict_['workflow']),
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
    _allowed_methods = ['run_again']
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "WorkflowRun Base Schema",
        "required": [],
        "properties": {
            "workflow": {
                "type": "object",
                "title": "Workflow",
                "classes": ["dessia_common.workflow.Workflow"],
                "order": 0,
                "editable": False,
                "description": "Workflow"
            },
            'output_value': {
                "type": "object",
                "classes": "Any",
                "title": "Values",
                "description": "Input and output values",
                "editable": False,
                "order": 1
            },
            'input_values': {
                'type': 'object',
                'order': 2,
                'editable': False,
                'title': 'Input Values',
                'patternProperties': {
                    '.*': {
                        'type': "object",
                        'classes': 'Any'
                    }
                }
            },
            'variables_values': {
                'type': 'object',
                'order': 3,
                'editable': False,
                'title': 'Variables Values',
                'patternProperties': {
                    '.*': {
                        'type': "object",
                        'classes': 'Any'
                    }
                }
            },
            'start_time': {
                "type": "number",
                "title": "Start Time",
                "editable": False,
                "description": "Start time of simulation",
                "order": 4
            },
            'end_time': {
                "type": "number",
                "title": "End Time",
                "editable": False,
                "description": "End time of simulation",
                "order": 5
            },
            'log': {
                "type": "string",
                "title": "Log",
                "editable": False,
                "description": "Log",
                "order": 6
            }
        }
    }

    def __init__(self, workflow, input_values, output_value, variables_values,
                 start_time, end_time, log, name=''):
        self.workflow = workflow
        self.input_values = input_values
        self.output_value = output_value
        self.variables_values = variables_values
        self.start_time = start_time
        self.end_time = end_time
        self.execution_time = end_time - start_time
        self.log = log

        dc.DessiaObject.__init__(self, name=name)

    def _data_eq(self, other_workflow_run):
        # TODO : Should we add input_values and variables values in test ?
        if dc.is_sequence(self.output_value):
            if not dc.is_sequence(other_workflow_run):
                return False
            equal_output = all([v == other_v for v, other_v
                                in zip(self.output_value,
                                       other_workflow_run.output_value)])
        else:
            equal_output = self.output_value == other_workflow_run.output_value
        return self.workflow == other_workflow_run.workflow and equal_output

    def _data_hash(self):
        # TODO : Should we add input_values and variables values in test ?
        if dc.is_sequence(self.output_value):
            hash_output = dc.list_hash(self.output_value)
        else:
            hash_output = hash(self.output_value)
        return hash(self.workflow) + int(hash_output % 10e5)

    def _display_angular(self):
        d_blocks = [b for b in self.workflow.blocks if hasattr(b, '_display')]
        sorted_d_blocks = sorted(d_blocks, key=lambda b: b.order)
        displays = self.workflow._display_angular()
        for block in sorted_d_blocks:
            local_values = {}
            for input_ in block.inputs:
                indices = self.workflow.variable_indices(input_)
                local_values[input_] = self.variables_values[str(indices)]
            display = block._display(local_values)
            displays.extend(display)
        if isinstance(self.output_value, dc.DessiaObject):
            displays.extend(self.output_value._display_angular())
        return displays

    @classmethod
    def dict_to_object(cls, dict_):
        workflow = Workflow.dict_to_object(dict_['workflow'])
        if 'output_value' in dict_ and 'output_value_type' in dict_:
            type_ = dict_['output_value_type']
            value = dict_['output_value']
            output_value = dc.recursive_instantiation(type_=type_, value=value)
        else:
            output_value = None

        input_values = {int(i): dc.deserialize(v)
                        for i, v in dict_['input_values'].items()}
        variables_values = {k: dc.deserialize(v)
                            for k, v in dict_['variables_values'].items()}
        return cls(workflow=workflow, output_value=output_value,
                   input_values=input_values,
                   variables_values=variables_values,
                   start_time=dict_['start_time'], end_time=dict_['end_time'],
                   log=dict_['log'], name=dict_['name'])

    def to_dict(self):
        input_values = {i: dc.serialize(v)
                        for i, v in self.input_values.items()}
        variables_values = {k: dc.serialize(v)
                            for k, v in self.variables_values.items()}
        dict_ = dc.DessiaObject.base_dict(self)
        dict_.update({'workflow': self.workflow.to_dict(),
                      'input_values': input_values,
                      'variables_values': variables_values,
                      'start_time': self.start_time,
                      'end_time': self.end_time,
                      'execution_time': self.execution_time,
                      'log': self.log})

        if self.output_value is not None:
            dict_.update({'output_value': dc.serialize(self.output_value),
                          'output_value_type': dc.recursive_type(self.output_value)})

        return dict_

    def dict_to_arguments(self, dict_):
        return self.workflow.dict_to_arguments(dict_=dict_)

    def method_dict(self, method_name, method_jsonschema):
        if method_name == 'run_again':
            dict_ = dc.serialize_dict(self.input_values)
            for property_, value in method_jsonschema['properties'].items():
                if property_ in dict_\
                        and 'object_id' in value\
                        and 'object_class' in value:
                    # TODO : Check. this is probably useless as we are not dealing with default values here
                    dict_[property_] = value
            return dict_
        return dc.DessiaObject.method_dict(method_name=method_name,
                                           jsonschema=method_jsonschema)

    def run_again(self, input_values):
        workflow_run = self.workflow.run(input_values=input_values,
                                         verbose=False,
                                         progress_callback=None,
                                         name=None)
        return workflow_run

    @property
    def _method_jsonschemas(self):
        # TODO : Share code with Workflow run method
        jsonschemas = {'run_again': deepcopy(dc.JSONSCHEMA_HEADER)}
        properties_dict = jsonschemas['run_again']['properties']
        required_inputs = []
        for i, value in self.input_values.items():
            current_dict = {}
            input_ = self.workflow.inputs[i]
            annotation = (str(i), input_.type_)
            input_block = self.workflow.block_from_variable(input_)
            if input_block.name:
                title = dc.prettyname(input_block.name + ' - ' + input_.name)
            else:
                title = dc.prettyname(input_.name)
            annotation_jsonschema = dc.jsonschema_from_annotation(
                annotation=annotation,
                jsonschema_element=current_dict,
                order=i,
                title=title,
            )
            current_dict.update(annotation_jsonschema[str(i)])
            if not isinstance(input_, (VariableWithDefaultValue,
                                       TypedVariableWithDefaultValue)):
                required_inputs.append(str(i))
            else:
                current_dict.update(dc.set_default_value(current_dict,
                                                         str(i),
                                                         input_.default_value))
            properties_dict[str(i)] = current_dict[str(i)]
        jsonschemas['run_again']['required'] = required_inputs
        return jsonschemas


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
        if argument not in ['self', 'progress_callback']:
            type_ = dc.type_from_annotation(method.__annotations__[argument],
                                            module=method.__module__)
            if iargument >= nargs - ndefault_args:
                default = args_specs.defaults[ndefault_args-nargs+iargument]
                input_ = TypedVariableWithDefaultValue(type_=type_,
                                                       default_value=default,
                                                       name=argument)
                inputs.append(input_)
                
            else:
                inputs.append(TypedVariable(type_=type_, name=argument))
    return inputs


def value_type_check(value, type_):
    # try:
    #     typeguard.check_type('', value, type_)
    # except TypeError:
    #     return False
    try:# TODO: Subscripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass
            
    return True
    