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
from typing import List, Union, Type, Any, Dict, Tuple, get_type_hints
from copy import deepcopy
from dessia_common.templates import workflow_template
import itertools
from dessia_common import DessiaObject, DisplayObject, Filter, is_sequence,\
    list_hash, serialize_typing, serialize, get_python_class_from_class_name,\
    deserialize, type_from_annotation, enhanced_deep_attr, is_bounded,\
    deprecation_warning, JSONSCHEMA_HEADER, jsonschema_from_annotation,\
    deserialize_argument, set_default_value, prettyname, dict_to_object,\
    serialize_dict, UntypedArgumentError, recursive_type,\
    recursive_instantiation, full_classname
from dessia_common.vectored_objects import ParetoSettings, from_csv
from dessia_common.typings import JsonSerializable
import warnings

# import plot_data

# Type Aliases
VariableTypes = Union['Variable', 'TypedVariable',
                      'VariableWithDefaultValue',
                      'TypedVariableWithDefaultValue']


# VariableTypes = Subclass['Variable']


class Variable(DessiaObject):
    _standalone_in_db = False
    _eq_is_data_eq = False
    has_default_value: bool = False

    def __init__(self, memorize: bool = False, name: str = ''):
        self.memorize = memorize
        DessiaObject.__init__(self, name=name)
        self.position = None

    def to_dict(self):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'has_default_value': self.has_default_value})
        return dict_


class TypedVariable(Variable):
    has_default_value: bool = False

    def __init__(self, type_: Type, memorize: bool = False, name: str = ''):
        Variable.__init__(self, memorize=memorize, name=name)
        self.type_ = type_

    def to_dict(self):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'type_': serialize_typing(self.type_),
                      'memorize': self.memorize,
                      'has_default_value': self.has_default_value})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        type_ = get_python_class_from_class_name(dict_['type'])
        memorize = dict_['memorize']
        return cls(type_=type_, memorize=memorize, name=dict_['name'])


class VariableWithDefaultValue(Variable):
    has_default_value: bool = True

    def __init__(self, default_value: Any, memorize: bool = False,
                 name: str = ''):
        Variable.__init__(self, memorize=memorize, name=name)
        self.default_value = default_value


class TypedVariableWithDefaultValue(TypedVariable):
    has_default_value: bool = True

    def __init__(self, type_: Type, default_value: Any,
                 memorize: bool = False, name: str = ''):
        TypedVariable.__init__(self, type_=type_, memorize=memorize, name=name)
        self.default_value = default_value

    def to_dict(self):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'type_': serialize_typing(self.type_),
                      'default_value': serialize(self.default_value),
                      'memorize': self.memorize,
                      'has_default_value': self.has_default_value})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        type_ = get_python_class_from_class_name(dict_['type_'])
        default_value = deserialize(dict_['default_value'])
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
            output_items = zip(dict_['output_names'], obj.outputs)
            for output_name, output_ in output_items:
                output_.name = output_name
        return obj

    return func_wrapper


class Block(DessiaObject):
    _standalone_in_db = False
    _eq_is_data_eq = False
    _non_serializable_attributes = []

    def __init__(self, inputs: List[VariableTypes],
                 outputs: List[VariableTypes],
                 position: Tuple[float, float] = None, name: str = ''):
        self.inputs = inputs
        self.outputs = outputs
        if position is None:
            self.position = (0, 0)
        else:
            self.position = position

        DessiaObject.__init__(self, name=name)

    def equivalent_hash(self):
        return len(self.__class__.__name__)

    def equivalent(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def to_dict(self):
        dict_ = DessiaObject.base_dict(self)
        dict_['inputs'] = [i.to_dict() for i in self.inputs]
        dict_['outputs'] = [o.to_dict() for o in self.outputs]
        if self.position is not None:
            dict_['position'] = list(self.position)
        else:
            dict_['position'] = self.position
        return dict_

    def jointjs_data(self):
        data = {'block_class': self.__class__.__name__}
        if self.name != '':
            data['name'] = self.name
        else:
            data['name'] = self.__class__.__name__
        return data


class Display(Block):
    # _jsonschema = {}
    _displayable_input = 0

    def __init__(self, inputs: List[VariableTypes] = None,
                 order: int = 0, name: str = ''):
        self.order = order
        if inputs is None:
            inputs = [TypedVariable(type_=DessiaObject,
                                    name='Model to Display', memorize=True)]

        Block.__init__(self, inputs=inputs, outputs=[], name=name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.order == other.order

    def equivalent_hash(self):
        return self.order

    def display_(self, local_values: Dict[VariableTypes, Any], **kwargs):
        object_ = local_values[self.inputs[self._displayable_input]]
        displays = object_._displays()
        return displays

    def to_dict(self):
        dict_ = DessiaObject.base_dict(self)
        dict_['order'] = self.order
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(order=dict_['order'], name=dict_['name'])

    @staticmethod
    def evaluate(self):
        return []


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

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.type_ == other.type_

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_['type_'] = self.type_
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(type_=dict_['type_'], name=dict_['name'])

    def evaluate(self, values):
        dirname = os.path.dirname(__file__)
        relative_filepath = 'models/data/' + values[self.inputs[0]]
        filename = os.path.join(dirname, relative_filepath)
        if self.type_ == 'csv':
            array, variables = from_csv(filename=filename, end=None,
                                        remove_duplicates=True)
            return [array, variables]
        msg = 'File type {} not supported'.format(self.type_)
        raise NotImplementedError(msg)


class InstantiateModel(Block):
    """
    :param model_class: The class to instanciate.
    :type model_class: DessiaObject
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, model_class: Type, name: str = ''):
        self.model_class = model_class
        inputs = []

        inputs = set_inputs_from_function(self.model_class.__init__, inputs)

        outputs = [TypedVariable(type_=self.model_class,
                                 name='Instanciated object')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.model_class.__name__)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        classname = self.model_class.__class__.__name__
        other_classname = other.model_class.__class__.__name__
        return classname == other_classname

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_['model_class'] = full_classname(object_=self.model_class,
                                              compute_for='class')
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        if 'model_class_module' in dict_:
            # TODO Retro-compatibility. Remove this in future versions
            module_name = dict_['model_class_module']
            classname = module_name + '.' + dict_['model_class']
        else:
            classname = dict_['model_class']
        class_ = get_python_class_from_class_name(classname)
        return cls(class_, name=dict_['name'])

    def evaluate(self, values):
        args = {var.name: values[var] for var in self.inputs}
        return [self.model_class(**args)]

    def package_mix(self):
        return {self.model_class.__module__.split('.')[0]: 1}


class InstanciateModel(InstantiateModel):
    def __init__(self, model_class: Type, name: str = ''):
        InstantiateModel.__init__(self, model_class=model_class, name=name)
        warnings.warn(
            "InstanciateModel is deprecated, use InstantiateModel instead",
            DeprecationWarning
        )


class ClassMethod(Block):
    def __init__(self, class_: Type,
                 method_name: str, name: str = ''):
        self.class_ = class_
        self.method_name = method_name
        inputs = []
        method = getattr(self.class_, self.method_name)
        inputs = set_inputs_from_function(method, inputs)

        self.argument_names = [i.name for i in inputs]

        annotations = get_type_hints(method)
        type_ = type_from_annotation(annotations['return'],
                                     method.__module__)
        output_name = 'method result of {}'.format(self.method_name)
        outputs = [TypedVariable(type_=type_, name=output_name)]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.class_.__name__) + 7 * len(self.method_name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        same_class = self.class_.__name__ == other.class_.__name__
        same_method = self.method_name == other.method_name
        return same_class and same_method

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'method_name': self.method_name,
                      'class_': full_classname(object_=self.class_,
                                               compute_for='class')})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        if 'class_module' in dict_:
            # TODO Retro-compatibility. Remove this in future versions
            module_name = dict_['class_module']
            classname = module_name + '.' + dict_['class_']
        else:
            classname = dict_['class_']
        class_ = get_python_class_from_class_name(classname)
        method_name = dict_['method_name']
        name = dict_['name']
        return cls(class_=class_, method_name=method_name, name=name)

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

    def __init__(self, model_class: Type, method_name: str, name: str = ''):
        self.model_class = model_class
        self.method_name = method_name
        inputs = [TypedVariable(type_=model_class, name='model at input')]
        method = getattr(self.model_class, self.method_name)

        inputs = set_inputs_from_function(method, inputs)

        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        result_output_name = 'method result of {}'.format(self.method_name)
        annotations = get_type_hints(method)
        if 'return' in annotations:
            type_ = type_from_annotation(annotations['return'],
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
        return len(self.model_class.__name__) + 7 * len(self.method_name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        same_model = self.model_class.__name__ == other.model_class.__name__
        same_method = self.method_name == other.method_name
        return same_model and same_method

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'method_name': self.method_name,
                      'model_class': full_classname(object_=self.model_class,
                                                    compute_for='class')})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        if 'model_class_module' in dict_:
            # TODO Retro-compatibility. Remove this in future versions
            module_name = dict_['model_class_module']
            classname = module_name + '.' + dict_['model_class']
        else:
            classname = dict_['model_class']
        class_ = get_python_class_from_class_name(classname)
        method_name = dict_['method_name']
        name = dict_['name']
        return cls(model_class=class_, method_name=method_name, name=name)

    def evaluate(self, values):
        args = {arg_name: values[var]
                for arg_name, var in zip(self.argument_names, self.inputs[1:])
                if var in values}
        return [getattr(values[self.inputs[0]], self.method_name)(**args),
                values[self.inputs[0]]]

    def package_mix(self):
        return {self.model_class.__module__.split('.')[0]: 1}


# class Function(Block):
#     def __init__(self, function: Callable, name: str = ''):
#         self.function = function
#         inputs = []
#         annotations = get_type_hints(function)
#         for arg_name in inspect.signature(function).parameters.keys():
#             # TODO: Check why we need TypedVariables
#             type_ = type_from_annotation(annotations[arg_name])
#             inputs.append(TypedVariable(type_=type_, name=arg_name))
#         out_type = type_from_annotation(annotations['return'])
#         outputs = [TypedVariable(type_=out_type, name='Output function')]
#
#         Block.__init__(self, inputs, outputs, name=name)
#
#     def equivalent_hash(self):
#         return int(hash(self.function.__name__) % 10e5)
#
#     def equivalent(self, other):
#         return self.function == other.function
#
#     def evaluate(self, values):
#         return self.function(*values)


class Sequence(Block):
    def __init__(self, number_arguments: int, name: str = ''):
        # type_: Subclass[DessiaObject] = None,
        self.number_arguments = number_arguments
        prefix = 'Sequence element {}'
        inputs = [Variable(name=prefix.format(i))
                  for i in range(self.number_arguments)]
        # if type_ is None:
        # else:
        #     inputs = [TypedVariable(type_=type_, name=prefix.format(i))
        #               for i in range(self.number_arguments)]

        # self.type_ = type_
        outputs = [TypedVariable(type_=list, name='sequence')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return self.number_arguments

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.number_arguments == other.number_arguments

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_['number_arguments'] = self.number_arguments
        # if self.type_ is not None:
        #     dict_['type_'] = serialize_typing(self.type_)
        # else:
        #     dict_['type_'] = None
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        # if dict_['type_'] is not None:
        #     type_ = deserialize_typing(dict_['type_'])
        # else:
        #     type_ = None
        return cls(dict_['number_arguments'], dict_['name'])

    def evaluate(self, values):
        return [[values[var] for var in self.inputs]]


class ForEach(Block):
    """
    :param workflow_block: The WorkflowBlock on which iterate.
    :type workflow_block: WorkflowBlock
    :param iter_input_index: Index of iterable input in worklow_block.inputs
    :type iter_input_index: int
    :param input_connections: Links ForEach's inputs to its
        workflow_block's inputs.
        input_connections[i] = [ForEach_input_j, WorkflowBlock_input_k]
    :type input_connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    :param output_connections: Same but for outputs.
        output_connections[i] = [WorkflowBlock_output_j, ForEach_output_k]
    :type output_connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, workflow_block: 'WorkflowBlock',
                 iter_input_index: int, name: str = ''):
        self.workflow_block = workflow_block
        self.iter_input_index = iter_input_index
        self.iter_input = self.workflow_block.inputs[iter_input_index]
        inputs = []
        for i, workflow_input in enumerate(self.workflow_block.inputs):
            if i == iter_input_index:
                name = 'Iterable input: ' + workflow_input.name
                inputs.append(Variable(name=name))
            else:
                input_ = workflow_input.copy()
                input_.name = 'binding ' + input_.name
                inputs.append(input_)
        output_variable = Variable(name='Foreach output')
        self.output_connections = None  # TODO: configuring port internal connections
        self.input_connections = None

        Block.__init__(self, inputs, [output_variable], name=name)

    def equivalent_hash(self):
        return int(self.workflow_block.equivalent_hash() % 10e5)

    def equivalent(self, other):
        # TODO Check this method. Is indices_eq mandatory ?
        if not Block.equivalent(self, other):
            return False
        workflow = self.workflow_block.workflow
        other_workflow = other.workflow_block.workflow

        indices = workflow.variable_indices(self.iter_input)
        other_indices = other_workflow.variable_indices(other.iter_input)

        same_workflow_block = self.workflow_block == other.workflow_block
        same_indices = indices == other_indices
        return same_workflow_block and same_indices

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'workflow_block': self.workflow_block.to_dict(),
                      'iter_input_index': self.iter_input_index})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        workflow_block = WorkflowBlock.dict_to_object(dict_['workflow_block'])
        iter_input_index = dict_['iter_input_index']
        return cls(workflow_block=workflow_block,
                   iter_input_index=iter_input_index, name=dict_['name'])

    def evaluate(self, values):
        values_workflow = {var2: values[var1]
                           for var1, var2 in zip(self.inputs,
                                                 self.workflow_block.inputs)}
        output_values = []
        for value in values_workflow[self.iter_input]:
            values_workflow[self.iter_input] = value
            output = self.workflow_block.evaluate(values_workflow)[0]
            output_values.append(output)
        return [output_values]


class Unpacker(Block):
    def __init__(self, indices: List[int], name: str = ''):
        self.indices = indices
        inputs = [Variable(name='input_sequence')]
        outputs = [Variable(name='output_{}'.format(i)) for i in indices]

        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.indices == other.indices

    def equivalent_hash(self):
        return len(self.indices)

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_['indices'] = self.indices
        return dict_

    @classmethod
    def dict_to_object(cls, dict_):
        return cls(dict_['indices'], dict_['name'])

    def evaluate(self, values):
        return [values[self.inputs[0]][i] for i in self.indices]


class Flatten(Block):
    def __init__(self, name: str = ''):
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


class Product(Block):
    def __init__(self, number_list: int, name=''):
        self.number_list = number_list
        inputs = []
        for i in range(self.number_list):
            inputs.append(Variable(name='list_product_' + str(i)))

        output_variable = Variable(name='Product output')
        Block.__init__(self, inputs, [output_variable], name=name)

    def equivalent_hash(self):
        return self.number_list

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False

        return self.number_list == other.number_list

    def to_dict(self):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'number_list': self.number_list})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        number_list = dict_['number_list']
        return cls(number_list=number_list, name=dict_['name'])

    def evaluate(self, values):
        list_product = [values[var] for var in self.inputs]
        output_value = list(itertools.product(*list_product))
        return [output_value]


class Filter(Block):
    """ 
    :param filters: A list of dictionaries,
                    each corresponding to a value to filter.
                    The dictionary should be as follows :
                    *{'attribute' : Name of attribute to filter (str),
                      'operator' : choose between gt, lt, get, let
                                  (standing for greater than, lower than,
                                   geater or equal than,
                                   lower or equal than) (str),
                      'bound' :  the value (float)}*
    :type filters: list[dict]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, filters: List[Filter], name: str = ''):
        self.filters = filters
        inputs = [Variable(name='input_list')]
        outputs = [Variable(name='output_list')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.filters == other.filters

    def equivalent_hash(self):
        hashes = [hash(v) for f in self.filters for v in f.values()]
        return int(sum(hashes) % 10e5)

    def to_dict(self):
        dict_ = Block.to_dict(self)
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
                value = enhanced_deep_attr(object_, filter_['attribute'])
                bounded = is_bounded(filter_, value)
                i += 1

            if bounded:
                ouput_values.append(object_)
        return [ouput_values]


class MultiPlot(Display):
    """
    :param attributes: A List of all attributes that will be shown inside the \
    ParallelPlot window on DessIA's Platform.
    :type attributes: List[str]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, attributes: List[str], order: int = 0, name: str = ''):
        self.attributes = attributes
        pareto_input = TypedVariableWithDefaultValue(type_=ParetoSettings,
                                                     default_value=None,
                                                     memorize=True,
                                                     name='Pareto settings')
        inputs = [Variable(memorize=True, name='input_list'), pareto_input]
        Display.__init__(self, inputs=inputs, order=order, name=name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False

        same_attributes = self.attributes == other.attributes
        same_order = self.order == other.order
        return same_attributes and same_order

    def equivalent_hash(self):
        return sum([len(a) for a in self.attributes]) + self.order

    def display_(self, local_values, **kwargs):
        import plot_data
        if 'reference_path' not in kwargs:
            reference_path = 'output_value'  # TODO bof bof bof
        else:
            reference_path = kwargs['reference_path']
        display_input = self.inputs[self._displayable_input]
        objects = local_values[display_input]
        # pareto_settings = local_values[self.inputs[1]]

        values = [{a: enhanced_deep_attr(o, a) for a in self.attributes}
                  for o in objects]

        first_vars = self.attributes[:2]
        values2d = [{key: val[key]} for key in first_vars for val in
                    values]

        tooltip = plot_data.Tooltip(name='Tooltip',
                                    to_disp_attribute_names=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip,
                                        to_disp_attribute_names=first_vars,
                                        elements=values2d,
                                        name='Scatter Plot')

        rgbs = [[192, 11, 11], [14, 192, 11], [11, 11, 192]]
        parallelplot = plot_data.ParallelPlot(
            disposition='horizontal', to_disp_attribute_names=self.attributes,
            rgbs=rgbs, elements=values
        )
        objects = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300),
                 plot_data.Window(width=560, height=300)]
        coords = [(0, 0), (0, 300)]
        multiplot = plot_data.MultiplePlots(elements=values, plots=objects,
                                            sizes=sizes, coords=coords,
                                            name='Results plot')
        display_ = DisplayObject(type_='plot_data', data=multiplot,
                                 reference_path=reference_path)
        return [display_.to_dict()]

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'attributes': self.attributes, 'order': self.order})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(attributes=dict_['attributes'],
                   order=dict_['order'],
                   name=dict_['name'])

    @staticmethod
    def evaluate(_):
        return []


class ParallelPlot(MultiPlot):
    """

    DEPRECATED. USE MULTIPLOT INSTEAD

    :param attributes: A List of all attributes that will be shown inside the \
    ParallelPlot window on the DessIA Platform.
    :type attributes: List[str]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, attributes: List[str], order: int = 0, name: str = ''):
        deprecation_warning(self.__class__.__name__, 'Class', 'MultiPlot')
        MultiPlot.__init__(self, attributes=attributes, order=order, name=name)

    def to_dict(self):
        dict_ = MultiPlot.to_dict(self)
        dict_.update({'object_class': 'dessia_common.workflow.MultiPlot'})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return MultiPlot(attributes=dict_['attributes'], order=dict_['order'],
                         name=dict_['name'])


class ModelAttribute(Block):
    """
    :param attribute_name: The name of the attribute to select.
    :type attribute_name: str
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, attribute_name: str, name: str = ''):
        self.attribute_name = attribute_name

        inputs = [Variable(name='Model')]
        outputs = [Variable(name='Model attribute')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.attribute_name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.attribute_name == other.attribute_name

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'attribute_name': self.attribute_name})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(dict_['attribute_name'], dict_['name'])

    def evaluate(self, values):
        return [enhanced_deep_attr(values[self.inputs[0]],
                                      self.attribute_name)]


class Sum(Block):
    def __init__(self, number_elements: int = 2, name: str = ''):
        self.number_elements = number_elements
        inputs = [Variable(name='Sum element {}'.format(i + 1))
                  for i in range(number_elements)]
        outputs = [Variable(name='Sum')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return self.number_elements

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.number_elements == other.number_elements

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'number_elements': self.number_elements})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(dict_['number_elements'], dict_['name'])

    @staticmethod
    def evaluate(values):
        return [sum(values)]


class Substraction(Block):
    def __init__(self, name: str = ''):
        inputs = [Variable(name='+'), Variable(name='-')]
        outputs = [Variable(name='Substraction')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return 0

    def equivalent(self, other):
        if not Block.equivalent(self, other):
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


class Pipe(DessiaObject):
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

    def __init__(self, input_variable: VariableTypes,
                 output_variable: VariableTypes, name: str = ''):
        self.input_variable = input_variable
        self.output_variable = output_variable

        DessiaObject.__init__(self, name=name)

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
        "required": ["blocks", "pipes", "outputs"],
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
            if pipe.input_variable not in self.variables:
                self.variables.append(pipe.input_variable)
                self.nonblock_variables.append(pipe.input_variable)
            if pipe.output_variable not in self.variables:
                self.variables.append(pipe.output_variable)
                self.nonblock_variables.append(pipe.output_variable)

        self._utd_graph = False

        input_variables = []

        for variable in self.variables:
            if (variable not in self.imposed_variable_values) and \
                    (len(nx.ancestors(self.graph, variable)) == 0):
                # !!! Why not just : if nx.ancestors(self.graph, variable) ?
                # if not hasattr(variable, 'type_'):
                #     msg = 'Workflow as an untyped input variable: {}'
                #     raise WorkflowError(msg.format(variable.name))
                input_variables.append(variable)

        Block.__init__(self, input_variables, [output], name=name)
        self.output = self.outputs[0]

    def _data_hash(self):
        base_hash = len(self.blocks) \
                    + 11 * len(self.pipes) \
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

        copied_workflow = Workflow(
            blocks=blocks, pipes=pipes, output=output,
            imposed_variable_values=imposed_variable_values, name=self.name
        )
        return copied_workflow

    def _displays(self) -> List[JsonSerializable]:
        display_object = DisplayObject(type_='workflow',
                                          data=self.to_dict())
        displays = [display_object.to_dict()]
        return displays

    @property
    def _method_jsonschemas(self):
        jsonschemas = {'run': deepcopy(JSONSCHEMA_HEADER)}
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
                    name = input_block.name + ' - ' + input_.name
                    title = prettyname(name)
                else:
                    title = prettyname(input_.name)

            annotation_jsonschema = jsonschema_from_annotation(
                annotation=annotation,
                jsonschema_element=current_dict,
                order=i,
                title=title
            )
            current_dict.update(annotation_jsonschema[str(i)])
            if not input_.has_default_value:
                required_inputs.append(str(i))
            else:
                dict_ = set_default_value(
                    jsonschema_element=current_dict, key=str(i),
                    default_value=input_.default_value
                )
                current_dict.update(dict_)
            properties_dict[str(i)] = current_dict[str(i)]
        jsonschemas['run']['required'] = required_inputs
        jsonschemas['run']['method'] = True
        return jsonschemas

    def to_dict(self):
        self.refresh_blocks_positions()
        dict_ = Block.to_dict(self)
        blocks = [b.to_dict() for b in self.blocks]
        pipes = []
        for pipe in self.pipes:
            pipes.append((self.variable_indices(pipe.input_variable),
                          self.variable_indices(pipe.output_variable)))

        dict_.update({'blocks': blocks, 'pipes': pipes,
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
    def dict_to_object(cls, dict_: JsonSerializable) -> 'Workflow':
        blocks = [DessiaObject.dict_to_object(d) for d in dict_['blocks']]
        if 'nonblock_variables' in dict_:
            nonblock_variables = [dict_to_object(d)
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

        if 'imposed_variable_values' in dict_ and 'imposed_variables' in dict_:
            imposed_variable_values = {}
            iterator = zip(dict_['imposed_variables'],
                           dict_['imposed_variable_values'])
            for variable_index, serialized_value in iterator:
                value = deserialize(serialized_value)
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

    def dict_to_arguments(self, dict_: JsonSerializable, method: str):
        if method in self._allowed_methods:
            arguments_values = {}
            for i, input_ in enumerate(self.inputs):
                has_default = input_.has_default_value
                if not has_default or (has_default and str(i) in dict_):
                    value = dict_[str(i)]
                    deserialized_value = deserialize_argument(
                        type_=input_.type_, argument=value
                    )
                    arguments_values[i] = deserialized_value

            arguments = {'input_values': arguments_values}
            return arguments
        msg = 'Method {} not in Workflow allowed methods'
        raise NotImplementedError(msg.format(method))

    @classmethod
    def variable_from_index(cls, index, blocks, nonblock_variables):
        """
        Index elements are, in order :
        - Block index : int
        - int representin port side (0: input, 1: output)
        - Port index : int
        """
        if type(index) == int:
            variable = nonblock_variables[index]
        else:
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
                return ib1, ti1, iv1
            if variable in block.outputs:
                ib1 = iblock
                ti1 = 1
                iv1 = block.outputs.index(variable)
                return ib1, ti1, iv1

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
            if block not in ancestors:
                disconnected_elements.append(block)

        for variable in self.nonblock_variables:
            if variable not in ancestors:
                disconnected_elements.append(variable)
        return disconnected_elements

    def index(self, variable):
        index = self.inputs.index(variable)
        return index

    def layout(self, min_horizontal_spacing=300, min_vertical_spacing=200,
               max_height=800, max_length=1500):
        coordinates = {}
        elements_by_distance = {}
        for element in self.blocks + self.nonblock_variables:
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
            max_distance = 3  # TODO: this is an awfull quick fix

        horizontal_spacing = max(min_horizontal_spacing,
                                 max_length / max_distance)

        for i, distance in enumerate(
                sorted(elements_by_distance.keys())[::-1]):
            n = len(elements_by_distance[distance])
            vertical_spacing = min(min_vertical_spacing, max_height / n)
            horizontal_anchor_size = max_distance
            for j, element in enumerate(elements_by_distance[distance]):
                coordinates[element] = (i * horizontal_spacing,
                                        (j + 0.5) * vertical_spacing)
        return coordinates

    def refresh_blocks_positions(self):
        coordinates = self.layout()
        for i, block in enumerate(self.blocks):
            block.position = coordinates[block]
        for i, nonblock in enumerate(self.nonblock_variables):
            nonblock.position = coordinates[nonblock]

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
            values[variable] = value
            activated_items[variable] = True

        # Input activation
        for index, variable in enumerate(self.inputs):
            if index in input_values:
                value = input_values[index]
                values[variable] = value
                activated_items[variable] = True
            elif hasattr(variable, 'default_value'):
                values[variable] = variable.default_value
                activated_items[variable] = True
            else:
                msg = 'Value {} of index {} in inputs has no value'
                raise ValueError(msg.format(variable.name, index))

        something_activated = True

        start_time = time.time()

        log_msg = 'Starting workflow run at {}'
        log_line = log_msg.format(time.strftime('%d/%m/%Y %H:%M:%S UTC',
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
                        values[pipe.output_variable] = values[
                            pipe.input_variable]
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
                            progress += 1 / len(self.blocks)
                            progress_callback(progress)

                        # Unpacking result of evaluation
                        output_items = zip(block.outputs, output_values)
                        for output, output_value in output_items:
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
            name = self.name + ' run'
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
                                      'has_default': hasattr(block, 'default')}
                                     for i in block.inputs],
                          'outputs': [{'name': o.name,
                                       'workflow_output': o in self.outputs}
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
            # TOCHECK Is it necessary to add is_workflow_input/output
            #  for outputs/inputs ??
            block_data = block.jointjs_data()
            inputs = [{'name': i.name,
                       'is_workflow_input': i in self.inputs,
                       'has_default_value': hasattr(i, 'default_value')}
                      for i in block.inputs]
            outputs = [{'name': o.name,
                        'is_workflow_output': o in self.outputs}
                       for o in block.outputs]
            block_data.update({'inputs': inputs,
                               'outputs': outputs,
                               'position': coordinates[block]})
            blocks.append(block_data)

        nonblock_variables = []
        for variable in self.nonblock_variables:
            is_input = variable in self.inputs
            nonblock_variables.append({'name': variable.name,
                                       'is_workflow_input': is_input,
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
        data = json.dumps(self.jointjs_data())
        rendered_template = workflow_template.substitute(workflow_data=data)

        temp_file = tempfile.mkstemp(suffix='.html')[1]
        with open(temp_file, 'wb') as file:
            file.write(rendered_template.encode('utf-8'))

        webbrowser.open('file://' + temp_file)

    def is_valid(self):
        # Checking types of each end of pipes
        for pipe in self.pipes:
            if hasattr(pipe.input_variable, 'type_') \
                    and hasattr(pipe.output_variable, 'type_'):
                type1 = pipe.input_variable.type_
                type2 = pipe.output_variable.type_
                if type1 != type2:
                    try:
                        consistent = issubclass(pipe.input_variable.type_,
                                                pipe.output_variable.type_)
                    except TypeError:
                        # TODO: need of a real typing check
                        consistent = True
                        if not consistent:
                            msg = """
                            Inconsistent pipe type from pipe input {}
                            to pipe output {}: {} incompatible with {}
                            """
                            raise TypeError(msg.format(
                                pipe.input_variable.name,
                                pipe.output_variable.name,
                                pipe.input_variable.type_,
                                pipe.output_variable.type_)
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
        return {pn: f / fraction_sum for pn, f in package_mix.items()}


class WorkflowBlock(Block):
    """
    Wrapper around workflow to put it in a block of another workflow
    Even if a workflow is a block, it can't be used directly as it has
    a different behavior
    than a Block in eq and hash which is problematic to handle in dicts
    for example

    :param workflow: The WorkflowBlock's workflow
    :type workflow: Workflow
    :param input_connections: Links ForEach's inputs to its workflow_block's inputs. input_connections[i] = [ForEach_input_j, WorkflowBlock_input_k]
    :type input_connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    :param output_connections: Same but for outputs. output_connections[i] = [WorkflowBlock_output_j, ForEach_output_k]
    :type output_connections: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    """

    def __init__(self, workflow: Workflow,
                 name: str = ''):
        self.workflow = workflow
        self.input_connections = None  # TODO: configuring port internal connections
        self.output_connections = None
        inputs = []
        for i, variable in enumerate(self.workflow.inputs):
            input_ = variable.copy()
            input_.name = '{} - {}'.format(name, variable.name)
            inputs.append(input_)

        outputs = [self.workflow.output.copy()]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return hash(self.workflow)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.workflow == other.workflow

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


class WorkflowRun(DessiaObject):
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

        DessiaObject.__init__(self, name=name)

    def _data_eq(self, other_workflow_run):
        # TODO : Should we add input_values and variables values in test ?
        if is_sequence(self.output_value):
            if not is_sequence(other_workflow_run):
                return False
            equal_output = all([v == other_v for v, other_v
                                in zip(self.output_value,
                                       other_workflow_run.output_value)])
        else:
            equal_output = self.output_value == other_workflow_run.output_value
        return self.workflow == other_workflow_run.workflow and equal_output

    def _data_hash(self):
        # TODO : Should we add input_values and variables values in test ?
        if is_sequence(self.output_value):
            hash_output = list_hash(self.output_value)
        else:
            hash_output = hash(self.output_value)
        return hash(self.workflow) + int(hash_output % 10e5)

    def _displays(self) -> List[JsonSerializable]:
        d_blocks = [b for b in self.workflow.blocks if hasattr(b, 'display_')]
        sorted_d_blocks = sorted(d_blocks, key=lambda b: b.order)
        displays = self.workflow._displays()
        for block in sorted_d_blocks:
            reference_path = ''
            local_values = {}
            for i, input_ in enumerate(block.inputs):
                strindices = str(self.workflow.variable_indices(input_))
                local_values[input_] = self.variables_values[strindices]
                if i == block._displayable_input:
                    reference_path = 'variables_values/' + strindices
            display = block.display_(local_values=local_values,
                                     reference_path=reference_path)
            displays.extend(display)
        if isinstance(self.output_value, DessiaObject):
            displays.extend(self.output_value._displays(
                reference_path='output_value'
            ))
        return displays

    @classmethod
    def dict_to_object(cls, dict_):
        workflow = Workflow.dict_to_object(dict_['workflow'])
        if 'output_value' in dict_ and 'output_value_type' in dict_:
            type_ = dict_['output_value_type']
            value = dict_['output_value']
            output_value = recursive_instantiation(type_=type_, value=value)
        else:
            output_value = None

        input_values = {int(i): deserialize(v)
                        for i, v in dict_['input_values'].items()}
        variables_values = {k: deserialize(v)
                            for k, v in dict_['variables_values'].items()}
        return cls(workflow=workflow, output_value=output_value,
                   input_values=input_values,
                   variables_values=variables_values,
                   start_time=dict_['start_time'], end_time=dict_['end_time'],
                   log=dict_['log'], name=dict_['name'])

    def to_dict(self):
        input_values = {i: serialize(v)
                        for i, v in self.input_values.items()}
        variables_values = {k: serialize(v)
                            for k, v in self.variables_values.items()}
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'workflow': self.workflow.to_dict(),
                      'input_values': input_values,
                      'variables_values': variables_values,
                      'start_time': self.start_time, 'end_time': self.end_time,
                      'execution_time': self.execution_time, 'log': self.log})

        if self.output_value is not None:
            dict_.update({
                'output_value': serialize(self.output_value),
                'output_value_type': recursive_type(self.output_value)
            })

        return dict_

    def dict_to_arguments(self, dict_: JsonSerializable, method: str):
        if method in self._allowed_methods:
            return self.workflow.dict_to_arguments(dict_=dict_, method='run')
        msg = 'Method {} not in WorkflowRun allowed methods'
        raise NotImplementedError(msg.format(method))

    def method_dict(self, method_name: str = None,
                    method_jsonschema: Any = None):
        if method_name is not None and method_name == 'run_again' \
                and method_jsonschema is not None:
            dict_ = serialize_dict(self.input_values)
            for property_, value in method_jsonschema['properties'].items():
                if property_ in dict_ \
                        and 'object_id' in value and 'object_class' in value:
                    # TODO : Check. this is probably useless as we are not dealing with default values here
                    dict_[property_] = value
            return dict_
        # TODO Check this result. Might raise an error
        return DessiaObject.method_dict(self, method_name=method_name,
                                           method_jsonschema=method_jsonschema)

    def run_again(self, input_values, progress_callback=None):
        workflow_run = self.workflow.run(input_values=input_values,
                                         verbose=False,
                                         progress_callback=progress_callback,
                                         name=None)
        return workflow_run

    @property
    def _method_jsonschemas(self):
        jsonschemas = self.workflow._method_jsonschemas
        jsonschemas['run_again'] = jsonschemas.pop('run')
        return jsonschemas


def set_inputs_from_function(method, inputs=None):
    """
    
    """
    if inputs is None:
        inputs = []
    args_specs = inspect.getfullargspec(method)
    nargs = len(args_specs.args) - 1

    if args_specs.defaults is not None:
        ndefault_args = len(args_specs.defaults)
    else:
        ndefault_args = 0

    for iarg, argument in enumerate(args_specs.args[1:]):
        if argument not in ['self', 'progress_callback']:
            try:
                annotations = get_type_hints(method)
                type_ = type_from_annotation(annotations[argument],
                                                module=method.__module__)
            except KeyError:
                msg = 'Argument {} of method/function {} has no typing'
                raise UntypedArgumentError(
                    msg.format(argument, method.__name__))
            if iarg >= nargs - ndefault_args:
                default = args_specs.defaults[ndefault_args - nargs + iarg]
                input_ = TypedVariableWithDefaultValue(type_=type_,
                                                       default_value=default,
                                                       name=argument)
                inputs.append(input_)

            else:
                inputs.append(TypedVariable(type_=type_, name=argument))
    return inputs


def value_type_check(value, type_):
    try:  # TODO: Subscripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass

    return True
