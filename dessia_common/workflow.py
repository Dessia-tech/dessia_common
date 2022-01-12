#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os
import inspect
import time
import tempfile
import json
import webbrowser
import networkx as nx
from typing import List, Union, Type, Any, Dict,\
    Tuple, Optional, get_type_hints
from copy import deepcopy
from dessia_common.templates import workflow_template
import itertools

from dessia_common import DessiaObject, DisplayObject, DessiaFilter, \
    is_sequence, list_hash, serialize, is_bounded, \
    type_from_annotation,\
    enhanced_deep_attr, deprecation_warning, JSONSCHEMA_HEADER,\
    jsonschema_from_annotation, deserialize_argument, set_default_value,\
    prettyname, serialize_dict,\
    recursive_type, recursive_instantiation
from dessia_common.errors import UntypedArgumentError
from dessia_common.utils.diff import data_eq
from dessia_common.utils.serialization import dict_to_object, deserialize,\
    serialize_with_pointers
from dessia_common.utils.types import get_python_class_from_class_name,\
    serialize_typing, full_classname, deserialize_typing
from dessia_common.utils.copy import deepcopy_value
from dessia_common.vectored_objects import from_csv
from dessia_common.typings import JsonSerializable, MethodType,\
    ClassMethodType
import warnings

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
        type_ = deserialize_typing(dict_['type_'])
        memorize = dict_['memorize']
        return cls(type_=type_, memorize=memorize, name=dict_['name'])

    def copy(self, deep: bool = False, memo=None):
        return TypedVariable(type_=self.type_, memorize=self.memorize,
                             name=self.name)


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

    def copy(self, deep: bool = False, memo=None):
        copied_default_value = deepcopy_value(self.default_value, memo=memo)
        return TypedVariableWithDefaultValue(type_=self.type_,
                                             default_value=copied_default_value,
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
    _displayable_input = 0
    _non_editable_attributes = ['inputs']

    def __init__(self, inputs: List[VariableTypes] = None, order: int = 0, name: str = ''):
        self.order = order
        if inputs is None:
            inputs = [Variable(name='Model to Display', memorize=True)]

        Block.__init__(self, inputs=inputs, outputs=[], name=name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.order == other.order

    def equivalent_hash(self):
        return self.order

    def display_(self, local_values: Dict[VariableTypes, Any], **kwargs):
        object_ = local_values[self.inputs[self._displayable_input]]
        displays = object_._displays(**kwargs)
        return displays

    def to_dict(self):
        dict_ = Block.to_dict(self)
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
    :type model_class: Instanciable
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
    def __init__(self, method_type: ClassMethodType[Type], name: str = ''):
        self.method_type = method_type
        inputs = []
        method = getattr(method_type.class_, method_type.name)
        inputs = set_inputs_from_function(method, inputs)

        self.argument_names = [i.name for i in inputs]

        output_name = 'method result of {}'.format(method_type.name)
        annotations = get_type_hints(method)
        if 'return' in annotations:
            type_ = type_from_annotation(annotations['return'],
                                         method.__module__)
            outputs = [TypedVariable(type_=type_, name=output_name)]
        else:
            outputs = [Variable(name=output_name)]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        classname = self.method_type.class_.__name__
        return len(classname) + 7 * len(self.method_type.name)

    def equivalent(self, other: 'ClassMethod'):
        if not Block.equivalent(self, other):
            return False
        classname = self.method_type.class_.__name__
        other_classname = other.method_type.class_.__name__
        same_class = classname == other_classname
        same_method = self.method_type.name == other.method_type.name
        return same_class and same_method

    def to_dict(self):
        dict_ = Block.to_dict(self)
        classname = full_classname(object_=self.method_type.class_,
                                   compute_for='class')
        method_type_dict = {'class_': classname, 'name': self.method_type.name}
        dict_.update({'method_type': method_type_dict})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_) -> 'ClassMethod':
        if 'method_type' in dict_:
            classname = dict_['method_type']['class_']
            method_name = dict_['method_type']['name']
        else:
            # Retro-compatibility
            classname = dict_['model_class']
            method_name = dict_['method_name']
        class_ = get_python_class_from_class_name(classname)
        name = dict_['name']
        method_type = ClassMethodType(class_=class_, name=method_name)
        return cls(method_type=method_type, name=name)

    def evaluate(self, values):
        args = {arg_name: values[var]
                for arg_name, var in zip(self.argument_names, self.inputs)
                if var in values}
        return [getattr(self.method_type.class_,
                        self.method_type.name)(**args)]


class ModelMethod(Block):
    """
    :param method_type: Represent class and method used.
    :type method_type: MethodType[T]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, method_type: MethodType[Type], name: str = ''):
        self.method_type = method_type
        inputs = [TypedVariable(type_=method_type.class_,
                                name='model at input')]
        method = getattr(method_type.class_, method_type.name)

        inputs = set_inputs_from_function(method, inputs)

        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        result_output_name = 'method result of {}'.format(method_type.name)
        annotations = get_type_hints(method)
        if 'return' in annotations:
            type_ = type_from_annotation(annotations['return'],
                                         method.__module__)
            return_output = TypedVariable(type_=type_, name=result_output_name)
        else:
            return_output = Variable(name=result_output_name)

        model_output_name = 'model at output {}'.format(method_type.name)
        model_output = TypedVariable(type_=method_type.class_,
                                     name=model_output_name)
        outputs = [return_output, model_output]
        if name == '':
            name = 'Model method: {}'.format(method_type.name)
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        classname = self.method_type.class_.__name__
        return len(classname) + 7 * len(self.method_type.name)

    def equivalent(self, other: 'ModelMethod'):
        if not Block.equivalent(self, other):
            return False
        classname = self.method_type.class_.__name__
        other_classname = other.method_type.class_.__name__
        same_model = classname == other_classname
        same_method = self.method_type.name == other.method_type.name
        return same_model and same_method

    def to_dict(self):
        dict_ = Block.to_dict(self)
        classname = full_classname(object_=self.method_type.class_,
                                   compute_for='class')
        method_type_dict = {'class_': classname, 'name': self.method_type.name}
        dict_.update({'method_type': method_type_dict})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_) -> 'ModelMethod':
        if 'method_type' in dict_:
            classname = dict_['method_type']['class_']
            method_name = dict_['method_type']['name']
        else:
            # Retro-compatibility
            classname = dict_['model_class']
            method_name = dict_['method_name']
        class_ = get_python_class_from_class_name(classname)
        name = dict_['name']
        method_type = MethodType(class_=class_, name=method_name)
        return cls(method_type=method_type, name=name)

    def evaluate(self, values):
        args = {arg_name: values[var]
                for arg_name, var in zip(self.argument_names, self.inputs[1:])
                if var in values}
        return [getattr(values[self.inputs[0]], self.method_type.name)(**args),
                values[self.inputs[0]]]

    def package_mix(self):
        return {self.method_type.class_.__module__.split('.')[0]: 1}


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
        wb_hash = int(self.workflow_block.equivalent_hash() % 10e5)
        return wb_hash + self.iter_input_index

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        input_eq = self.iter_input_index == other.iter_input_index
        wb_eq = self.workflow_block.equivalent(other.workflow_block)
        return wb_eq and input_eq

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
    :type filters: list[DessiaFilter]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, filters: List[DessiaFilter], name: str = ''):
        self.filters = filters
        inputs = [Variable(name='input_list')]
        outputs = [Variable(name='output_list')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.filters == other.filters

    def equivalent_hash(self):
        hashes = [hash(f) for f in self.filters]
        return int(sum(hashes) % 10e5)

    def to_dict(self):
        dict_ = Block.to_dict(self)
        dict_.update({'filters': [f.to_dict() for f in self.filters]})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls([DessiaFilter.dict_to_object(d) for d in dict_['filters']], dict_['name'])

    def evaluate(self, values):
        ouput_values = []
        objects = values[self.inputs[0]]
        for object_ in objects:
            bounded = True
            i = 0
            while bounded and i < len(self.filters):
                filter_ = self.filters[i]
                value = enhanced_deep_attr(object_, filter_.attribute)
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
        Display.__init__(self, order=order, name=name)
        self.inputs[0].name = 'Input List'

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

        tooltip = plot_data.Tooltip(name='Tooltip', attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip,
                                        x_variable=first_vars[0],
                                        y_variable=first_vars[1],
                                        elements=values2d,
                                        name='Scatter Plot')

        rgbs = [[192, 11, 11], [14, 192, 11], [11, 11, 192]]
        parallelplot = plot_data.ParallelPlot(
            disposition='horizontal', axes=self.attributes,
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
        "python_typing": 'dessia_common.workflow.Pipe',
        "standalone_in_db": False,
        "required": ["input_variable", "output_variable"],
        "properties": {
            "input_variable": {
                "type": "object", "editable": True, "order": 0,
                "python_typing": "List[dessia_common.workflow.VariableTypes]",
                "classes": [
                    "dessia_common.workflow.Variable",
                    "dessia_common.workflow.TypedVariable",
                    "dessia_common.workflow.VariableWithDefaultValue",
                    "dessia_common.workflow.TypedVariableWithDefaultValue"
                ]
            },
            "output_variable": {
                "type": "object", "editable": True, "order": 1,
                "python_typing": "List[dessia_common.workflow.VariableTypes]",
                "classes": [
                    "dessia_common.workflow.Variable",
                    "dessia_common.workflow.TypedVariable",
                    "dessia_common.workflow.VariableWithDefaultValue",
                    "dessia_common.workflow.TypedVariableWithDefaultValue"
                ],
            },
            "name": {
                'type': 'string', 'title': 'Name', 'editable': True,
                'order': 2, 'default_value': '',
                'python_typing': 'builtins.str'
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
    _eq_is_data_eq = True

    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "Workflow",
        "required": ["blocks", "pipes", "outputs"],
        "python_typing": 'dessia_common.workflow.Pipe',
        "standalone_in_db": True,
        "properties": {
            "blocks": {
                "type": "array", "order": 0, "editable": True,
                "python_typing": "SubclassOf[dessia_common.workflow.Block]",
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
                "python_typing": "List[dessia_common.workflow.Pipe]",
                "items": {
                    'type': 'objects',
                    'classes': ["dessia_common.workflow.Pipe"],
                    "python_type": "dessia_common.workflow.Pipe",
                    "editable": True
                }
            },
            "outputs": {
                "type": "array", "order": 2,
                "python_typing": "List[dessia_common.workflow.VariableTypes]",
                'items': {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'python_typing': "dessia_common.workflow.VariableTypes"
                }
            },
            "name": {
                'type': 'string', 'title': 'Name', 'editable': True,
                'order': 3, 'default_value': '',
                'python_typing': 'builtins.str'
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
            upstream_var = pipe.input_variable
            downstream_var = pipe.output_variable
            if upstream_var not in self.variables:
                self.variables.append(upstream_var)
                self.nonblock_variables.append(upstream_var)
            if downstream_var not in self.variables:
                self.variables.append(downstream_var)
                self.nonblock_variables.append(downstream_var)

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
        output_hash = self.variable_indices(self.outputs[0])
        if not isinstance(output_hash, int):
            output_hash = sum(output_hash)

        base_hash = len(self.blocks) \
            + 11 * len(self.pipes) \
            + output_hash
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

    def is_valid(self):

        return True

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        blocks = [b.__deepcopy__() for b in self.blocks]
        copied_workflow = Workflow(
            blocks=blocks, pipes=[], output=None,
            name=self.name
        )

        pipes = []
        for pipe in self.pipes:
            input_index = self.variable_indices(pipe.input_variable)

            if isinstance(input_index, int):
                pipe_input = pipe.input_variable.copy()
            elif isinstance(input_index, tuple):
                pipe_input = copied_workflow.variable_from_index(input_index)
            else:
                raise ValueError("Could not find variable at index {}".format(input_index))

            output_index = self.variable_indices(pipe.output_variable)
            pipe_output = copied_workflow.variable_from_index(output_index)
            # print(input_index, output_index)
            copied_pipe = Pipe(pipe_input, pipe_output)
            # memo[pipe] = copied_pipe

            pipes.append(copied_pipe)

        # copied_workflow.pipes = pipes

        output = copied_workflow.variable_from_index(self.variable_indices(self.output))
        # copied_workflow.output = output

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            new_variable = copied_workflow.variable_from_index(self.variable_indices(variable))
            imposed_variable_values[new_variable] = value
        # copied_workflow.imposed_variable_values = imposed_variable_values

        copied_workflow = Workflow(
            blocks=blocks, pipes=pipes, output=output,
            imposed_variable_values=imposed_variable_values,
            name=self.name
        )
        return copied_workflow

    def _displays(self) -> List[JsonSerializable]:
        display_object = DisplayObject(type_='workflow', data=self.to_dict())
        displays = [display_object.to_dict()]
        return displays

    @property
    def _method_jsonschemas(self):
        jsonschemas = {'run': deepcopy(JSONSCHEMA_HEADER)}
        jsonschemas['run'].update({
            'classes': ['dessia_common.workflow.Workflow']
        })
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
                annotation=annotation, title=title,
                order=i + 1, jsonschema_element=current_dict,
            )
            # Order is i+1 because of name that is at 0
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
        properties_dict[str(len(self.inputs) + 1)] = {
            'type': 'string', 'title': 'WorkflowRun Name', 'editable': True,
            'order': 0, 'default_value': '', 'python_typing': 'builtins.str'
        }
        jsonschemas['run']['required'] = required_inputs
        jsonschemas['run']['method'] = True
        jsonschemas['run']['python_typing'] = serialize_typing(MethodType)
        return jsonschemas

    def to_dict(self, memo=None, use_pointers=True, path='#'):
        if memo is None:
            memo = {}

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

        # imposed_variables = []
        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            var_index = self.variable_indices(variable)
            # imposed_variables.append(var_index)

            if use_pointers:
                path_value = '{}/{}'.format(path, var_index)
                ser_value = serialize_with_pointers(value, memo=memo, path=path_value)

            else:
                ser_value = serialize(value)
            imposed_variable_values[var_index] = ser_value
            # if hasattr(value, 'to_dict'):
            #     imposed_variable_values.append(value.to_dict(memopath=''))
            # else:
            #     imposed_variable_values.append(value)

        # dict_['imposed_variables'] = imposed_variables
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
            if isinstance(source, int):
                variable1 = nonblock_variables[source]
            else:
                ib1, _, ip1 = source
                variable1 = blocks[ib1].outputs[ip1]

            if isinstance(target, int):
                variable2 = nonblock_variables[target]
            else:
                ib2, _, ip2 = target
                variable2 = blocks[ib2].inputs[ip2]

            pipes.append(Pipe(variable1, variable2))

        output = blocks[dict_['output'][0]].outputs[dict_['output'][2]]
        temp_workflow = cls(blocks=blocks, pipes=pipes, output=output)

        if 'imposed_variable_values' in dict_ and 'imposed_variables' in dict_:
            # Legacy support of double list
            imposed_variable_values = {}
            iterator = zip(dict_['imposed_variables'],
                           dict_['imposed_variable_values'])
            for variable_index, serialized_value in iterator:
                value = deserialize(serialized_value)
                variable = temp_workflow.variable_from_index(variable_index)
                # if type(variable_index) == int:
                #     variable = nonblock_variables[variable_index]
                # else:
                #     iblock, side, iport = variable_index
                #     if side:
                #         variable = blocks[iblock].outputs[iport]
                #     else:
                #         variable = blocks[iblock].inputs[iport]

                imposed_variable_values[variable] = value
        elif 'imposed_variable_values' in dict_:
            # New format with a dict
            imposed_variable_values = {}
            for variable_index, serialized_value in dict_['imposed_variable_values']:
                value = deserialize(serialized_value)
                variable = temp_workflow.variable_from_index(variable_index)
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

            name_index = len(self.inputs) + 1
            if str(name_index) in dict_:
                name = dict_[str(name_index)]
            else:
                name = None
            arguments = {'input_values': arguments_values, 'name': name}
            return arguments
        msg = 'Method {} not in Workflow allowed methods'
        raise NotImplementedError(msg.format(method))

    def method_dict(self, method_name: str = None,
                    method_jsonschema: Any = None):
        return {}

    def variable_from_index(self, index: Union[int, Tuple[int, int, int]]):
        """
        Index elements are, in order :
        - Block index : int
        - int representin port side (0: input, 1: output)
        - Port index : int
        """
        if isinstance(index, int):
            variable = self.nonblock_variables[index]
        else:
            if not index[1]:
                variable = self.blocks[index[0]].inputs[index[2]]
            else:
                variable = self.blocks[index[0]].outputs[index[2]]
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

        upstream_variable = self.get_upstream_nbv(variable)
        # Free variable not attached to block
        if upstream_variable in self.nonblock_variables:
            # If an upstream nbv is found, get its index
            return self.nonblock_variables.index(upstream_variable)

        msg = 'Something is wrong with variable {}'.format(variable.name)
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
        warnings.warn(
            "index method is deprecated, use input_index instead",
            DeprecationWarning
        )
        index = self.input_index(variable)
        return index

    def input_index(self, variable: VariableTypes) -> int:
        upstream_variable = self.get_upstream_nbv(variable)
        return self.inputs.index(upstream_variable)

    def variable_index(self, variable: VariableTypes) -> int:
        return self.variables.index(variable)

    def get_upstream_nbv(self, variable: VariableTypes) -> VariableTypes:
        """
        If given variable has an upstream nonblock_variable, return it
        otherwise return given variable itself
        """
        if not self.nonblock_variables:
            return variable
        incoming_pipes = [p for p in self.pipes
                          if p.output_variable == variable]
        if incoming_pipes:
            # Inputs can only be connected to one pipe
            incoming_pipe = incoming_pipes[0]
            if incoming_pipe.input_variable in self.nonblock_variables:
                return incoming_pipe.input_variable
        return variable

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
            progress_callback=lambda x: None,
            name=None):
        log = ''

        state = self.start_run(input_values)
        state.activate_inputs(check_all_inputs=True)

        start_time = time.time()

        log_msg = 'Starting workflow run at {}'
        log_line = log_msg.format(time.strftime('%d/%m/%Y %H:%M:%S UTC',
                                                time.gmtime(start_time)))
        log += (log_line + '\n')
        if verbose:
            print(log_line)

        state.continue_run(progress_callback=progress_callback)

        end_time = time.time()
        log_line = 'Workflow terminated in {} s'.format(end_time - start_time)

        log += log_line + '\n'
        if verbose:
            print(log_line)

        state.output_value = state.values[self.outputs[0]]

        if not name:
            name = self.name + ' run'
        return state.to_workflow_run(name=name)

    def start_run(self, input_values):
        activated_items = {p: False for p in self.pipes}
        activated_items.update({v: False for v in self.variables})
        activated_items.update({b: False for b in self.blocks})

        values = {}
        variables_values = {}
        return WorkflowState(self, input_values, activated_items, values,
                             variables_values, start_time=time.time())

    # def manual_run(self, input_values, name:str=''):
    #     log = ''

    #     start_time = time.time()

    #     log_msg = 'Starting manual workflow run at {}'
    #     log_line = log_msg.format(time.strftime('%d/%m/%Y %H:%M:%S UTC',
    #                                             time.gmtime(start_time)))
    #     log += (log_line + '\n')

    #     variable_values = {}
    #     for index_input, value in input_values.items():
    #         variable_values[self.variable_indices(self.inputs[index_input])] = value

    #     return ManualWorkflowRun(workflow=self, input_values=input_values,
    #                              output_value=None,
    #                              variables_values=variable_values,
    #                              evaluated_blocks = [],
    #                              log=log, name=name)

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
            if isinstance(input_index, int):
                node1 = input_index
            else:
                ib1, is1, ip1 = input_index
                if is1:
                    block = self.blocks[ib1]
                    ip1 += len(block.inputs)

                node1 = [ib1, ip1]

            output_index = self.variable_indices(pipe.output_variable)
            if isinstance(output_index, int):
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
        self.plot_jointjs(warn=False)

    def plot_jointjs(self, warn: bool = True):
        if warn:
            warnings.warn("Directly calling plot_jointjs is deprecated.\n"
                          "Please use plot instead.")
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
    """

    def __init__(self, workflow: Workflow,
                 name: str = ''):
        self.workflow = workflow
        # TODO: configuring port internal connections
        self.input_connections = None
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


class WorkflowState(DessiaObject):
    _standalone_in_db = True
    _allowed_methods = ['block_evaluation', 'evaluate_next_block',
                        'evaluate_maximum_blocks', 'add_block_input_values']
    _non_serializable_attributes = ['activated_items']

    def __init__(self, workflow: Workflow, input_values, activated_items,
                 values, variables_values, start_time, output_value=None,
                 log: str = '', name: str = ''):
        self.workflow = workflow
        self.input_values = input_values
        self.output_value = output_value
        self.variables_values = variables_values
        self.values = values
        self.start_time = start_time
        self.log = log

        self.activated_items = activated_items

        DessiaObject.__init__(self, name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        if not use_pointers:
            msg = 'WorkflowState to_dict should not' \
                  'be called with use_pointers=False'
            raise NotImplementedError(msg)
        if memo is None:
            memo = {}

        dict_ = DessiaObject.to_dict(self)

        values = {self.workflow.variable_index(i): serialize(v)
                  for i, v in self.values.items()}
        dict_.update({'values': values})
        dict_['evaluated_blocks_indices'] = [i for i, b
                                             in enumerate(self.workflow.blocks)
                                             if b in self.activated_items
                                             and self.activated_items[b]]
        dict_['evaluated_pipes_indices'] = [i for i, p
                                            in enumerate(self.workflow.pipes)
                                            if p in self.activated_items
                                            and self.activated_items[p]]
        dict_['evaluated_variables_indices'] = [
            self.workflow.variable_indices(v) for v in self.workflow.variables
            if v in self.activated_items and self.activated_items[v]
        ]
        if self.output_value is not None:
            dict_.update({
                'output_value': serialize(self.output_value),
                'output_value_type': recursive_type(self.output_value)
            })
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable,
                       force_generic: bool = False) -> 'WorkflowState':
        workflow = Workflow.dict_to_object(dict_['workflow'])
        if 'output_value' in dict_ and 'output_value_type' in dict_:
            type_ = dict_['output_value_type']
            value = dict_['output_value']
            output_value = recursive_instantiation(type_=type_, value=value)
        else:
            output_value = None

        values = {workflow.variables[int(i)]: deserialize(v)
                  for i, v in dict_['values'].items()}
        input_values = {int(i): deserialize(v)
                        for i, v in dict_['input_values'].items()}
        variables_values = {k: deserialize(v)
                            for k, v in dict_['variables_values'].items()}

        activated_items = {
            b: (True if i in dict_['evaluated_blocks_indices'] else False)
            for i, b in enumerate(workflow.blocks)
        }

        activated_items.update({
            p: (True if i in dict_['evaluated_pipes_indices'] else False)
            for i, p in enumerate(workflow.pipes)
        })

        var_indices = dict_['evaluated_variables_indices']
        activated_items.update({
            v: (True if workflow.variable_indices(v) in var_indices else False)
            for v in workflow.variables
        })

        return cls(workflow=workflow, input_values=input_values,
                   activated_items=activated_items, values=values,
                   variables_values=variables_values,
                   start_time=dict_['start_time'], output_value=output_value,
                   log=dict_['log'], name=dict_['name'])

    def add_input_value(self, input_index: int, value: Any):
        # TODO: Type checking?
        self.input_values[input_index] = value
        self.activate_inputs()

    def add_several_input_values(self, indices: List[int],
                                 values: Dict[str, Any]):
        for i in indices:
            self.add_input_value(input_index=i, value=values[str(i)])

    def add_block_input_values(self, block_index: int,
                               values: Dict[str, Any]):
        block = self.workflow.blocks[block_index]
        indices = [self.workflow.input_index(i) for i in block.inputs]
        self.add_several_input_values(indices=indices, values=values)

    def _displays(self) -> List[JsonSerializable]:
        data = self.to_dict()

        display_object = DisplayObject(type_='workflow_state', data=data)
        displays = [display_object.to_dict()]
        return displays

    @property
    def progress(self):
        activated_items = [b for b in self.workflow.blocks
                           if b in self.activated_items
                           and self.activated_items[b]]
        return len(activated_items) / len(self.workflow.blocks)

    def block_evaluation(self, block_index: int,
                         progress_callback=lambda x: None) -> bool:
        """
        Select a block to evaluate
        """
        block = self.workflow.blocks[block_index]

        self.activate_inputs()
        for pipe in self._activable_pipes():
            self._evaluate_pipe(pipe)

        if block in self._activable_blocks():
            self._evaluate_block(block)
            progress_callback(self.progress)
            return True
        else:
            return False

    def evaluate_next_block(self, progress_callback=lambda x: None) -> Optional[Block]:
        """
        Evaluate a block
        """
        self.activate_inputs()
        for pipe in self._activable_pipes():
            self._evaluate_pipe(pipe)

        blocks = self._activable_blocks()
        if blocks:
            block = blocks[0]
            self._evaluate_block(block)
            progress_callback(self.progress)
            return block
        else:
            return None

    def continue_run(self, progress_callback=lambda x: None):
        """
        Evaluate all possible blocks
        """
        self.activate_inputs()

        evaluated_blocks = []
        something_activated = True
        while something_activated:
            something_activated = False

            for pipe in self._activable_pipes():
                self._evaluate_pipe(pipe)
                something_activated = True

            for block in self._activable_blocks():
                evaluated_blocks.append(block)
                self._evaluate_block(block)
                progress_callback(self.progress)
                something_activated = True
        return evaluated_blocks

    def _activable_pipes(self):
        pipes = []
        for pipe in self.workflow.pipes:
            if not self.activated_items[pipe]:
                if self.activated_items[pipe.input_variable]:
                    pipes.append(pipe)
        return pipes

    def _activable_blocks(self):
        blocks = []
        for block in self.workflow.blocks:
            if not self.activated_items[block]:
                all_inputs_activated = True
                for function_input in block.inputs:
                    if not self.activated_items[function_input]:
                        all_inputs_activated = False
                        break

                if all_inputs_activated:
                    blocks.append(block)
        return blocks

    def _evaluate_pipe(self, pipe):
        self.activated_items[pipe] = True
        self.values[pipe.output_variable] = self.values[
            pipe.input_variable]
        self.activated_items[pipe.output_variable] = True

    def _evaluate_block(self, block, progress_callback=lambda x: x,
                        verbose=False):
        if verbose:
            log_line = 'Evaluating block {}'.format(block.name)
            self.log += log_line + '\n'
            if verbose:
                print(log_line)

        output_values = block.evaluate({i: self.values[i]
                                        for i in block.inputs})
        for input_ in block.inputs:
            if input_.memorize:
                indices = str(self.workflow.variable_indices(input_))  # Str is strange
                self.variables_values[indices] = self.values[input_]
        # Updating progress
        if progress_callback is not None:
            progress_callback(self.progress)

        # Unpacking result of evaluation
        output_items = zip(block.outputs, output_values)
        for output, output_value in output_items:
            if output.memorize:
                indices = str(self.workflow.variable_indices(output))
                self.variables_values[indices] = output_value
            self.values[output] = output_value
            self.activated_items[output] = True

        self.activated_items[block] = True

    def activate_inputs(self, check_all_inputs=False):
        """
        Returns if all inputs are activated
        """
        # Imposed variables values activation
        for variable, value in self.workflow.imposed_variable_values.items():
            # Type checking
            value_type_check(value, variable.type_)
            self.values[variable] = value
            self.activated_items[variable] = True

        # Input activation
        for index, variable in enumerate(self.workflow.inputs):
            if index in self.input_values:
                value = self.input_values[index]
                self.values[variable] = value
                self.activated_items[variable] = True
            elif hasattr(variable, 'default_value'):
                self.values[variable] = variable.default_value
                self.activated_items[variable] = True
            elif check_all_inputs:
                msg = 'Value {} of index {} in inputs has no value'
                if isinstance(variable, TypedVariable):
                    msg += ': should be instance of {}'.format(variable.type_)
                raise ValueError(msg.format(variable.name, index))

    def to_workflow_run(self, name=''):
        if self.progress == 1:
            return WorkflowRun(workflow=self.workflow,
                               input_values=self.input_values,
                               output_value=self.output_value,
                               variables_values=self.variables_values,
                               start_time=self.start_time, end_time=time.time(),
                               log=self.log, name=name)
        else:
            raise ValueError('Workflow not completed')


class WorkflowRun(DessiaObject):
    _standalone_in_db = True
    _allowed_methods = ['run_again']
    _eq_is_data_eq = True
    _jsonschema = {
        "definitions": {},
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": "WorkflowRun Base Schema",
        "required": [],
        "python_typing": 'dessia_common.workflow.Pipe',
        "standalone_in_db": True,
        "properties": {
            "workflow": {
                "type": "object", "title": "Workflow",
                "python_typing": "dessia_common.workflow.Workflow",
                "classes": ["dessia_common.workflow.Workflow"],
                "order": 0, "editable": False, "description": "Workflow"
            },
            'output_value': {
                "type": "object", "classes": "Any", "title": "Values",
                "description": "Input and output values",
                "editable": False, "order": 1, "python_typing": "Any"
            },
            'input_values': {
                'type': 'object', 'order': 2, 'editable': False,
                'title': 'Input Values', "python_typing": "Dict[str, Any]",
                'patternProperties': {
                    '.*': {
                        'type': "object",
                        'classes': 'Any'
                    }
                }
            },
            'variables_values': {
                'type': 'object', 'order': 3, 'editable': False,
                'title': 'Variables Values', "python_typing": "Dict[str, Any]",
                'patternProperties': {
                    '.*': {
                        'type': "object",
                        'classes': 'Any'
                    }
                }
            },
            'start_time': {
                "type": "number", "title": "Start Time",
                "editable": False, "python_typing": "builtins.int",
                "description": "Start time of simulation", "order": 4
            },
            'end_time': {
                "type": "number", "title": "End Time",
                "editable": False, "python_typing": "builtins.int",
                "description": "End time of simulation", "order": 5
            },
            'log': {
                "type": "string", "title": "Log",
                "editable": False, 'python_typing': 'builtins.str',
                "description": "Log", "order": 6,
            },
            "name": {
                'type': 'string', 'title': 'Name', 'editable': True,
                'order': 7, 'default_value': '',
                'python_typing': 'builtins.str'
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
        if not data_eq(self.output_value, other_workflow_run.output_value):
            return False
        return self.workflow._data_eq(other_workflow_run.workflow)

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

    # @classmethod
    # def dict_to_object(cls, dict_):
    #     workflow = Workflow.dict_to_object(dict_['workflow'])
    #     if 'output_value' in dict_ and 'output_value_type' in dict_:
    #         type_ = dict_['output_value_type']
    #         value = dict_['output_value']
    #         output_value = recursive_instantiation(type_=type_, value=value)
    #     else:
    #         output_value = None

    #     input_values = {int(i): deserialize(v)
    #                     for i, v in dict_['input_values'].items()}
    #     variables_values = {k: deserialize(v)
    #                         for k, v in dict_['variables_values'].items()}
    #     return cls(workflow=workflow, output_value=output_value,
    #                input_values=input_values,
    #                variables_values=variables_values,
    #                start_time=dict_['start_time'], end_time=dict_['end_time'],
    #                log=dict_['log'], name=dict_['name'])

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        input_values = {}

        if not use_pointers:
            raise NotImplementedError('WorkflowRun to_dict should not be called with use_pointers=False')

        if memo is None:
            memo = {}

        for i, v in self.input_values.items():
            serialized_v, memo = serialize_with_pointers(v, memo, path='#/input_values/{}'.format(i))
            input_values[i] = serialized_v

        variables_values = {}
        for k, v in self.variables_values.items():
            serialized_v, memo = serialize_with_pointers(v, memo,
                                                         path='#/variables_values/{}'.format(k))
            variables_values[k] = serialized_v

        # variables_values = {k: serialize(v)
        #                     for k, v in self.variables_values.items()}
        workflow = self.workflow.to_dict(path='#/workflow', memo=memo)

        dict_ = DessiaObject.base_dict(self)
        dict_.update({'workflow': workflow,
                      'input_values': input_values,
                      'variables_values': variables_values,
                      'start_time': self.start_time, 'end_time': self.end_time,
                      'execution_time': self.execution_time, 'log': self.log})

        if self.output_value is not None:
            serialized_output, _ = serialize_with_pointers(self.output_value, memo, path='#/output_value')
            dict_.update({
                'output_value': serialized_output,
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

    def run_again(self, input_values, progress_callback=None, name=None):
        workflow_run = self.workflow.run(input_values=input_values,
                                         verbose=False,
                                         progress_callback=progress_callback,
                                         name=name)
        return workflow_run

    @property
    def _method_jsonschemas(self):
        jsonschemas = self.workflow._method_jsonschemas
        jsonschemas['run_again'] = jsonschemas.pop('run')
        workflow_run_class = "dessia_common.workflow.WorkflowRun"
        jsonschemas['run_again']['classes'] = [workflow_run_class]
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
