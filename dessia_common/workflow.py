#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gathers all workflow relative features
"""
import os
import inspect
import time
import tempfile
import json
import webbrowser
from zipfile import ZipFile
import io
import networkx as nx
from typing import List, Union, Type, Any, Dict, Tuple, Optional, get_type_hints
from copy import deepcopy
from dessia_common.templates import workflow_template
import itertools
from dessia_common import DessiaObject, DisplayObject, DessiaFilter, is_sequence, is_bounded,\
    type_from_annotation, enhanced_deep_attr, split_argspecs, JSONSCHEMA_HEADER, jsonschema_from_annotation,\
    deserialize_argument, set_default_value, prettyname, serialize_dict
from dessia_common.errors import UntypedArgumentError
from dessia_common.utils.serialization import dict_to_object, deserialize, serialize_with_pointers, serialize,\
                                              dereference_jsonpointers
from dessia_common.utils.types import get_python_class_from_class_name, serialize_typing, full_classname,\
    deserialize_typing, recursive_type
from dessia_common.utils.copy import deepcopy_value
from dessia_common.utils.docstrings import parse_docstring, EMPTY_PARSED_ATTRIBUTE, FAILED_ATTRIBUTE_PARSING
from dessia_common.utils.diff import choose_hash
from dessia_common.vectored_objects import from_csv
from dessia_common.typings import JsonSerializable, MethodType, ClassMethodType
from dessia_common.files import StringFile, BinaryFile
import warnings

# Type Aliases
VariableTypes = Union['Variable', 'TypedVariable', 'VariableWithDefaultValue', 'TypedVariableWithDefaultValue']


class Variable(DessiaObject):
    _standalone_in_db = False
    _eq_is_data_eq = False
    has_default_value: bool = False

    def __init__(self, memorize: bool = False, name: str = ''):
        """
        Variable for workflow
        """
        self.memorize = memorize
        DessiaObject.__init__(self, name=name)
        self.position = None

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'has_default_value': self.has_default_value})
        return dict_


class TypedVariable(Variable):
    has_default_value: bool = False

    def __init__(self, type_: Type, memorize: bool = False, name: str = ''):
        """
        Variable for workflow with a typing
        """
        Variable.__init__(self, memorize=memorize, name=name)
        self.type_ = type_

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'type_': serialize_typing(self.type_), 'memorize': self.memorize,
                      'has_default_value': self.has_default_value})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None) -> 'TypedVariable':
        type_ = deserialize_typing(dict_['type_'])
        memorize = dict_['memorize']
        return cls(type_=type_, memorize=memorize, name=dict_['name'])

    def copy(self, deep: bool = False, memo=None):
        return TypedVariable(type_=self.type_, memorize=self.memorize, name=self.name)


class VariableWithDefaultValue(Variable):
    has_default_value: bool = True

    def __init__(self, default_value: Any, memorize: bool = False, name: str = ''):
        """
        A variable with a default value
        """
        Variable.__init__(self, memorize=memorize, name=name)
        self.default_value = default_value


class TypedVariableWithDefaultValue(TypedVariable):
    has_default_value: bool = True

    def __init__(self, type_: Type, default_value: Any, memorize: bool = False, name: str = ''):
        """
        Workflow variables wit a type and a default value
        """
        TypedVariable.__init__(self, type_=type_, memorize=memorize, name=name)
        self.default_value = default_value

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'type_': serialize_typing(self.type_), 'memorize': self.memorize,
                      'default_value': serialize(self.default_value), 'has_default_value': self.has_default_value})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None) -> 'TypedVariableWithDefaultValue':
        type_ = deserialize_typing(dict_['type_'])
        default_value = deserialize(dict_['default_value'], global_dict=global_dict,
                                    pointers_memo=pointers_memo)
        return cls(type_=type_,
                   default_value=default_value,
                   memorize=dict_['memorize'],
                   name=dict_['name'])

    def copy(self, deep: bool = False, memo=None):
        """
        :param deep: DESCRIPTION, defaults to False
        :type deep: bool, optional
        :param memo: a memo to use, defaults to None
        :type memo: TYPE, optional
        :return: The copied object
        """
        copied_default_value = deepcopy_value(self.default_value, memo=memo)
        return TypedVariableWithDefaultValue(type_=self.type_, default_value=copied_default_value,
                                             memorize=self.memorize, name=self.name)


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

    def __init__(self, inputs: List[VariableTypes], outputs: List[VariableTypes],
                 position: Tuple[float, float] = (0, 0), name: str = ''):
        """
        An Abstract block. Do not instantiate alone
        """
        self.inputs = inputs
        self.outputs = outputs
        self.position = position
        DessiaObject.__init__(self, name=name)

    def equivalent_hash(self):
        """
        Custom hash version of block that does not overwrite __hash__ as we do not want to lose
        python default equality behavior. Used by workflow module only.
        """
        return len(self.__class__.__name__)

    def equivalent(self, other):
        """
        Custom eq version of block that does not overwrite __eq__ as we do not want to lose
        python default equality behavior. Used by workflow module only.
        """
        return self.__class__.__name__ == other.__class__.__name__

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
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

    @staticmethod
    def _docstring():
        """
        Base function for submodel docstring computing
        """
        return None


class Display(Block):
    _displayable_input = 0
    _non_editable_attributes = ['inputs']

    def __init__(self, inputs: List[VariableTypes] = None, order: int = 0, name: str = ''):
        """
        Abstract class for display behaviors
        """
        self.order = order
        if inputs is None:
            inputs = [Variable(name='Model to Display', memorize=True)]
        Block.__init__(self, inputs=inputs, outputs=[], name=name)

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.order == other.order

    def equivalent_hash(self):
        return self.order

    def display_(self, local_values: Dict[VariableTypes, Any], **kwargs):
        object_ = local_values[self.inputs[self._displayable_input]]
        displays = object_._displays(**kwargs)
        return displays

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['order'] = self.order
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(order=dict_['order'], name=dict_['name'])

    @staticmethod
    def evaluate(self):
        return []


class Import(Block):
    def __init__(self, type_: str, name: str = ''):
        """
        Block that enables file imports
        """
        self.type_ = type_
        inputs = [TypedVariable(type_=str, name='Input filename'),
                  TypedVariableWithDefaultValue(type_=bool, default_value=True, name='Remove duplicates')]
        outputs = [Variable(name='Array'), Variable(name='Variables')]
        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

    def equivalent_hash(self):
        return len(self.type_)

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.type_ == other.type_

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['type_'] = self.type_
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(type_=dict_['type_'], name=dict_['name'])

    def evaluate(self, values):
        dirname = os.path.dirname(__file__)
        relative_filepath = 'models/data/' + values[self.inputs[0]]
        filename = os.path.join(dirname, relative_filepath)
        if self.type_ == 'csv':
            array, variables = from_csv(filename=filename, end=None, remove_duplicates=True)
            return [array, variables]
        raise NotImplementedError(f"File type {self.type_} not supported")


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
        outputs = [TypedVariable(type_=self.model_class, name='Instanciated object')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return len(self.model_class.__name__)

    def equivalent(self, other):
        classname = self.model_class.__class__.__name__
        other_classname = other.model_class.__class__.__name__
        return Block.equivalent(self, other) and classname == other_classname

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['model_class'] = full_classname(object_=self.model_class, compute_for='class')
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        if 'model_class_module' in dict_:  # TODO Retro-compatibility. Remove this in future versions
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

    def _docstring(self):
        docstring = self.model_class.__doc__
        annotations = get_type_hints(self.model_class.__init__)
        parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        parsed_attributes = parsed_docstring["attributes"]
        block_docstring = {i: parsed_attributes[i.name] if i.name in parsed_attributes
                           else EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring


class ClassMethod(Block):
    def __init__(self, method_type: ClassMethodType[Type], name: str = ''):
        self.method_type = method_type
        inputs = []

        self.method = method_type.get_method()
        inputs = set_inputs_from_function(self.method, inputs)

        self.argument_names = [i.name for i in inputs]
        output_name = f"method result of {method_type.name}"
        output = output_from_function(function=self.method, name=output_name)
        Block.__init__(self, inputs, [output], name=name)

    def equivalent_hash(self):
        classname = self.method_type.class_.__name__
        return len(classname) + 7 * len(self.method_type.name)

    def equivalent(self, other: 'ClassMethod'):
        classname = self.method_type.class_.__name__
        other_classname = other.method_type.class_.__name__
        same_class = classname == other_classname
        same_method = self.method_type.name == other.method_type.name
        return Block.equivalent(self, other) and same_class and same_method

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        classname = full_classname(object_=self.method_type.class_, compute_for='class')
        method_type_dict = {'class_': classname, 'name': self.method_type.name}
        dict_.update({'method_type': method_type_dict})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'ClassMethod':
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
        args = {arg_name: values[var] for arg_name, var in zip(self.argument_names, self.inputs) if var in values}
        return [self.method(**args)]

    def _docstring(self):
        method = self.method_type.get_method()
        docstring = method.__doc__
        annotations = get_type_hints(method)
        parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        parsed_attributes = parsed_docstring["attributes"]
        block_docstring = {i: parsed_attributes[i.name] if i.name in parsed_attributes
                           else EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring


class ModelMethod(Block):
    """
    :param method_type: Represent class and method used.
    :type method_type: MethodType[T]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, method_type: MethodType[Type], name: str = ''):
        self.method_type = method_type
        inputs = [TypedVariable(type_=method_type.class_, name='model at input')]
        method = method_type.get_method()
        inputs = set_inputs_from_function(method, inputs)

        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        return_output_name = f"method result of {method_type.name}"
        return_output = output_from_function(function=method, name=return_output_name)

        model_output_name = f"model at output {method_type.name}"
        model_output = TypedVariable(type_=method_type.class_, name=model_output_name)
        outputs = [return_output, model_output]
        if name == "":
            name = f"Model method: {method_type.name}"
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        classname = self.method_type.class_.__name__
        return len(classname) + 7 * len(self.method_type.name)

    def equivalent(self, other: 'ModelMethod'):
        classname = self.method_type.class_.__name__
        other_classname = other.method_type.class_.__name__
        same_model = classname == other_classname
        same_method = self.method_type.name == other.method_type.name
        return Block.equivalent(self, other) and same_model and same_method

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        classname = full_classname(object_=self.method_type.class_, compute_for='class')
        method_type_dict = {'class_': classname, 'name': self.method_type.name}
        dict_.update({'method_type': method_type_dict})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'ModelMethod':
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
        args = {arg_name: values[var] for arg_name, var in zip(self.argument_names, self.inputs[1:]) if var in values}
        return [getattr(values[self.inputs[0]], self.method_type.name)(**args), values[self.inputs[0]]]

    def package_mix(self):
        return {self.method_type.class_.__module__.split('.')[0]: 1}

    def _docstring(self):
        method = self.method_type.get_method()
        docstring = method.__doc__
        annotations = get_type_hints(method)
        parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        parsed_attributes = parsed_docstring["attributes"]
        block_docstring = {i: parsed_attributes[i.name] if i.name in parsed_attributes
                           else EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring


class Sequence(Block):
    def __init__(self, number_arguments: int, name: str = ''):
        self.number_arguments = number_arguments
        inputs = [Variable(name=f"Sequence element {i}") for i in range(self.number_arguments)]
        outputs = [TypedVariable(type_=list, name='sequence')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return self.number_arguments

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.number_arguments == other.number_arguments

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['number_arguments'] = self.number_arguments
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['number_arguments'], dict_['name'])

    def evaluate(self, values):
        return [[values[var] for var in self.inputs]]


class ForEach(Block):
    """
    A block to iterate on an input and perform an parralel for (iterations are not dependant)
    :param workflow_block: The WorkflowBlock on which iterate.
    :type workflow_block: WorkflowBlock
    :param iter_input_index: Index of iterable input in worklow_block.inputs
    :type iter_input_index: int
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, workflow_block: 'WorkflowBlock', iter_input_index: int, name: str = ''):
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
        input_eq = self.iter_input_index == other.iter_input_index
        wb_eq = self.workflow_block.equivalent(other.workflow_block)
        return Block.equivalent(self, other) and wb_eq and input_eq

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_.update({'workflow_block': self.workflow_block.to_dict(), 'iter_input_index': self.iter_input_index})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        workflow_block = WorkflowBlock.dict_to_object(dict_['workflow_block'])
        iter_input_index = dict_['iter_input_index']
        return cls(workflow_block=workflow_block, iter_input_index=iter_input_index, name=dict_['name'])

    def evaluate(self, values):
        values_workflow = {var2: values[var1] for var1, var2 in zip(self.inputs, self.workflow_block.inputs)}
        output_values = []
        for value in values_workflow[self.iter_input]:
            values_workflow[self.iter_input] = value
            output = self.workflow_block.evaluate(values_workflow)[0]
            output_values.append(output)
        return [output_values]

    def _docstring(self):
        wb_docstring = self.workflow_block._docstring()
        block_docstring = {}
        for i, inputs in enumerate(zip(self.inputs, self.workflow_block.workflow.inputs)):
            input_, workflow_input = inputs
            block_docstring[input_] = wb_docstring[workflow_input]
        return block_docstring


class Unpacker(Block):
    def __init__(self, indices: List[int], name: str = ''):
        self.indices = indices
        outputs = [Variable(name=f"output_{i}") for i in indices]
        Block.__init__(self, inputs=[Variable(name="input_sequence")], outputs=outputs, name=name)

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.indices == other.indices

    def equivalent_hash(self):
        return len(self.indices)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['indices'] = self.indices
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Unpacker':
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
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Flatten':
        return cls(dict_['name'])

    def evaluate(self, values):
        output = []
        for value in values[self.inputs[0]]:
            output.extend(value)
        return [output]


class Product(Block):
    def __init__(self, number_list: int, name: str = ''):
        self.number_list = number_list
        inputs = [Variable(name='list_product_' + str(i)) for i in range(self.number_list)]
        output_variable = Variable(name='Product output')
        Block.__init__(self, inputs, [output_variable], name=name)

    def equivalent_hash(self):
        return self.number_list

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.number_list == other.number_list

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'number_list': self.number_list})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        number_list = dict_['number_list']
        return cls(number_list=number_list, name=dict_['name'])

    def evaluate(self, values):
        """
        Computes the block: use itertools.product
        """
        list_product = [values[var] for var in self.inputs]
        output_value = list(itertools.product(*list_product))
        return [output_value]


class Filter(Block):
    """
    :param filters: A list of dictionaries, each corresponding to a value to filter.
                    The dictionary should be as follows :
                    *{'attribute' : Name of attribute to filter (str),
                      'operator' : choose between gt, lt, get, let (standing for greater than, lower than,
                                                                    geater or equal than, lower or equal than) (str),
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
        return Block.equivalent(self, other) and self.filters == other.filters

    def equivalent_hash(self):
        hashes = [hash(f) for f in self.filters]
        return int(sum(hashes) % 10e5)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_.update({'filters': [f.to_dict() for f in self.filters]})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls([DessiaFilter.dict_to_object(d) for d in dict_['filters']], dict_['name'])

    def evaluate(self, values):
        ouput_values = []
        for object_ in values[self.inputs[0]]:
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
    :param attributes: A List of all attributes that will be shown inside the ParallelPlot window on DessIA's Platform.
    :type attributes: List[str]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, attributes: List[str], order: int = 0, name: str = ''):
        self.attributes = attributes
        Display.__init__(self, order=order, name=name)
        self.inputs[0].name = 'Input List'

    def equivalent(self, other):
        same_attributes = self.attributes == other.attributes
        same_order = self.order == other.order
        return Block.equivalent(self, other) and same_attributes and same_order

    def equivalent_hash(self):
        return sum([len(a) for a in self.attributes]) + self.order

    def display_(self, local_values, **kwargs):
        import plot_data
        if 'reference_path' not in kwargs:
            reference_path = 'output_value'  # TODO bof bof bof
        else:
            reference_path = kwargs['reference_path']
        objects = local_values[self.inputs[self._displayable_input]]
        values = [{a: enhanced_deep_attr(o, a) for a in self.attributes} for o in objects]
        values2d = [{key: val[key]} for key in self.attributes[:2] for val in values]
        tooltip = plot_data.Tooltip(name='Tooltip', attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip, x_variable=self.attributes[0], y_variable=self.attributes[1],
                                        elements=values2d, name='Scatter Plot')

        parallelplot = plot_data.ParallelPlot(disposition='horizontal', axes=self.attributes,
                                              rgbs=[(192, 11, 11), (14, 192, 11), (11, 11, 192)], elements=values)
        objects = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=values, plots=objects, sizes=sizes,
                                            coords=[(0, 0), (0, 300)], name='Results plot')
        display_ = DisplayObject(type_='plot_data', data=multiplot, reference_path=reference_path)
        return [display_.to_dict()]

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_.update({'attributes': self.attributes, 'order': self.order})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(attributes=dict_['attributes'], order=dict_['order'], name=dict_['name'])

    @staticmethod
    def evaluate(_):
        return []


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
        return Block.equivalent(self, other) and self.attribute_name == other.attribute_name

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_.update({'attribute_name': self.attribute_name})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['attribute_name'], dict_['name'])

    def evaluate(self, values):
        return [enhanced_deep_attr(values[self.inputs[0]], self.attribute_name)]


class Sum(Block):
    def __init__(self, number_elements: int = 2, name: str = ''):
        self.number_elements = number_elements
        inputs = [Variable(name=f"Sum element {i + 1}") for i in range(number_elements)]
        Block.__init__(self, inputs=inputs, outputs=[Variable(name='Sum')], name=name)

    def equivalent_hash(self):
        return self.number_elements

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.number_elements == other.number_elements

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_.update({'number_elements': self.number_elements})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['number_elements'], dict_['name'])

    @staticmethod
    def evaluate(values):
        return [sum(values)]


class Substraction(Block):
    def __init__(self, name: str = ''):
        Block.__init__(self, [Variable(name='+'), Variable(name='-')], [Variable(name='Substraction')], name=name)

    def evaluate(self, values):
        return [values[self.inputs[0]] - values[self.inputs[1]]]


class Export(Block):
    def __init__(self, method_type: MethodType, export_name: str = "", name: str = ""):
        self.method_type = method_type
        if not export_name:
            export_name = "export"
        self.export_name = export_name

        method = method_type.get_method()

        self.extension = ""
        self.text = None

        output = output_from_function(function=method, name="export_output")
        Block.__init__(self, inputs=[TypedVariable(type_=method_type.class_)], outputs=[output], name=name)

    def evaluate(self, values):
        res = getattr(values[self.inputs[0]], self.method_type.name)()
        return [res]

    def _export_format(self, block_index: int):
        args = {"block_index": block_index}
        return {"extension": self.extension, "method_name": "export", "text": self.text, "args": args}


class ExportJson(Export):
    def __init__(self, model_class: Type, export_name: str = "", name: str = ""):
        self.model_class = model_class
        method_type = MethodType(class_=model_class, name="save_to_stream")

        Export.__init__(self, method_type=method_type, export_name=export_name, name=name)
        if not export_name:
            self.export_name += "_json"
        self.extension = "json"

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['model_class'] = full_classname(object_=self.method_type.class_, compute_for='class')
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        class_ = get_python_class_from_class_name(dict_['model_class'])
        return cls(class_, name=dict_['name'])


class ExportExcel(Export):
    def __init__(self, model_class: Type, export_name: str = "", name: str = ""):
        self.model_class = model_class
        method_type = MethodType(class_=model_class, name="to_xlsx_stream")

        Export.__init__(self, method_type=method_type, export_name=export_name, name=name)
        if not export_name:
            self.export_name += "_xlsx"
        self.extension = "xlsx"

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['model_class'] = full_classname(object_=self.method_type.class_, compute_for='class')
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        class_ = get_python_class_from_class_name(dict_['model_class'])
        return cls(class_, name=dict_['name'])


class Archive(Block):
    def __init__(self, number_exports: int = 1, name=""):
        self.number_exports = number_exports
        inputs = [Variable(name="export_" + str(i)) for i in range(number_exports)]
        Block.__init__(self, inputs=inputs, outputs=[Variable(name="zip archive")], name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_['number_exports'] = len(self.inputs)
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(number_exports=dict_["number_exports"], name=dict_['name'])

    def evaluate(self, values):
        archive = io.BytesIO()
        with ZipFile(archive, 'w') as zip_archive:
            for i, input_ in enumerate(self.inputs):
                value = values[input_]
                filename = f"file{i}.{value.extension}"
                if isinstance(value, StringFile):
                    with zip_archive.open(filename, 'w') as file:
                        file.write(value.getvalue().encode('utf-8'))
                elif isinstance(value, BinaryFile):
                    with zip_archive.open(filename, 'w') as file:
                        file.write(value.getbuffer())
                else:
                    raise ValueError("Archive input is not a file-like object")
        return [archive]

    @staticmethod
    def _export_format(block_index: int):
        return {"extension": "zip", "method_name": "export", "text": False, "args": {"block_index": block_index}}


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
        "definitions": {}, "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
        "title": "Pipe", "python_typing": 'dessia_common.workflow.Pipe', "standalone_in_db": False,
        "required": ["input_variable", "output_variable"],
        "properties": {
            "input_variable": {
                "type": "object", "editable": True, "order": 0,
                "python_typing": "List[dessia_common.workflow.VariableTypes]",
                "classes": ["dessia_common.workflow.Variable", "dessia_common.workflow.TypedVariable",
                            "dessia_common.workflow.VariableWithDefaultValue",
                            "dessia_common.workflow.TypedVariableWithDefaultValue"]
            },
            "output_variable": {
                "type": "object", "editable": True, "order": 1,
                "python_typing": "List[dessia_common.workflow.VariableTypes]",
                "classes": ["dessia_common.workflow.Variable", "dessia_common.workflow.TypedVariable",
                            "dessia_common.workflow.VariableWithDefaultValue",
                            "dessia_common.workflow.TypedVariableWithDefaultValue"],
            },
            "name": {'type': 'string', 'title': 'Name', 'editable': True,
                     'order': 2, 'default_value': '', 'python_typing': 'builtins.str'}
        }
    }
    _eq_is_data_eq = False

    def __init__(self, input_variable: VariableTypes, output_variable: VariableTypes, name: str = ''):
        self.input_variable = input_variable
        self.output_variable = output_variable
        DessiaObject.__init__(self, name=name)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        return {'input_variable': self.input_variable, 'output_variable': self.output_variable}


class WorkflowError(Exception):
    pass


class Workflow(Block):
    """
    :param blocks: A List with all the Blocks used by the Worklow.
    :type blocks: List[Block]
    :param pipes: A List of Pipe objects.
    :type pipes: List[Pipe]
    :param imposed_variable_values: A dictionary of imposed variable values.
    :type imposed_variable_values: Dict
    :param description: A short description that will be displayed on workflow card (frontend).
                        Should be shorter than 100 chars
    :type description: str
    :param documentation: A long documentation that will be displayed on workflow page (frontend).
                          Can use markdown elements.
    :param name: The name of the workflow.
    :type name: str
    """
    _standalone_in_db = True
    _allowed_methods = ['run', 'start_run']
    _eq_is_data_eq = True
    _jsonschema = {
        "definitions": {}, "$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "title": "Workflow",
        "required": ["blocks", "pipes", "outputs"], "python_typing": 'dessia_common.workflow.Pipe',
        "standalone_in_db": True,
        "properties": {
            "blocks": {
                "type": "array", "order": 0, "editable": True,
                "python_typing": "SubclassOf[dessia_common.workflow.Block]",
                "items": {"type": "object", "editable": True,
                          "classes": ["dessia_common.workflow.InstanciateModel", "dessia_common.workflow.ModelMethod",
                                      "dessia_common.workflow.ForEach", "dessia_common.workflow.ModelAttribute",
                                      "dessia_common.workflow.Function", "dessia_common.workflow.Sequence",
                                      "dessia_common.workflow.ForEach", "dessia_common.workflow.Unpacker",
                                      "dessia_common.workflow.Flatten", "dessia_common.workflow.Filter",
                                      "dessia_common.workflow.ParallelPlot", "dessia_common.workflow.Sum",
                                      "dessia_common.workflow.Substraction"]},
            },
            "pipes": {
                "type": "array", "order": 1, "editable": True, "python_typing": "List[dessia_common.workflow.Pipe]",
                "items": {
                    'type': 'objects', 'classes': ["dessia_common.workflow.Pipe"],
                    "python_type": "dessia_common.workflow.Pipe", "editable": True
                }
            },
            "outputs": {
                "type": "array", "order": 2, "python_typing": "List[dessia_common.workflow.VariableTypes]",
                'items': {
                    'type': 'array', 'items': {'type': 'number'},
                    'python_typing': "dessia_common.workflow.VariableTypes"
                }
            },
            "description": {"type": "string", "title": "Description", "editable": True,
                            "default_value": "", "python_typing": "builtins.str"},
            "documentation": {"type": "string", "title": "Documentation", "editable": True,
                              "default_value": "", "python_typing": "builtins.str"},
            "name": {'type': 'string', 'title': 'Name', 'editable': True, 'order': 3,
                     'default_value': '', 'python_typing': 'builtins.str'}
        }
    }

    def __init__(self, blocks, pipes, output, *, imposed_variable_values=None,
                 description: str = "", documentation: str = "", name: str = ""):
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
                raise ValueError(f"Cannot serialize block {block} ({block.name})")

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
                #     raise WorkflowError('Workflow as an untyped input variable: {}'.format(variable.name))
                input_variables.append(variable)

        self.description = description
        self.documentation = documentation

        output.memorize = True

        Block.__init__(self, input_variables, [output], name=name)
        self.output = self.outputs[0]

    def _data_hash(self):
        output_hash = self.variable_indices(self.outputs[0])
        if not isinstance(output_hash, int):
            output_hash = sum(output_hash)

        base_hash = len(self.blocks) + 11 * len(self.pipes) + output_hash
        block_hash = int(sum([b.equivalent_hash() for b in self.blocks]) % 10e5)
        return (base_hash + block_hash) % 1000000000

    def _data_eq(self, other_object):  # TODO: implement imposed_variable_values in equality
        if hash(self) != hash(other_object) or not Block.equivalent(self, other_object):
            return False
        # TODO: temp , reuse graph!!!!
        for block1, block2 in zip(self.blocks, other_object.blocks):
            if not block1.equivalent(block2):
                return False
        return True

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        blocks = [b.__deepcopy__() for b in self.blocks]
        output_adress = self.variable_indices(self.output)
        output_block = blocks[output_adress[0]]
        output = output_block.outputs[output_adress[2]]
        copied_workflow = Workflow(blocks=blocks, pipes=[], output=output, name=self.name)

        pipes = [self.copy_pipe(pipe=p, copied_workflow=copied_workflow) for p in self.pipes]

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            new_variable = copied_workflow.variable_from_index(self.variable_indices(variable))
            imposed_variable_values[new_variable] = value

        copied_workflow = Workflow(blocks=blocks, pipes=pipes, output=output,
                                   imposed_variable_values=imposed_variable_values, name=self.name)
        return copied_workflow

    def copy_pipe(self, pipe: Pipe, copied_workflow: 'Workflow') -> Pipe:
        upstream_index = self.variable_indices(pipe.input_variable)

        if isinstance(upstream_index, int):
            pipe_upstream = pipe.input_variable.copy()
        elif isinstance(upstream_index, tuple):
            pipe_upstream = copied_workflow.variable_from_index(upstream_index)
        else:
            raise ValueError(f"Could not find variable at index {upstream_index}")

        downstream_index = self.variable_indices(pipe.output_variable)
        pipe_downstream = copied_workflow.variable_from_index(downstream_index)
        return Pipe(pipe_upstream, pipe_downstream)

    def _displays(self) -> List[JsonSerializable]:
        displays = []
        documentation = self.to_markdown()
        if documentation.data:
            displays.append(documentation.to_dict())
        workflow = DisplayObject(type_='workflow', data=self.to_dict())
        displays.append(workflow.to_dict())
        return displays

    def to_markdown(self):
        """
        Sets workflow documentation as markdown
        """
        return DisplayObject(type_="markdown", data=self.documentation)

    def _docstring(self):
        """
        Computes documentation of all blocks
        """
        docstrings = [b._docstring() for b in self.blocks]
        return docstrings

    @property
    def _method_jsonschemas(self):
        jsonschemas = {'run': deepcopy(JSONSCHEMA_HEADER)}
        jsonschemas['run'].update({'classes': ['dessia_common.workflow.Workflow']})
        properties_dict = jsonschemas['run']['properties']
        required_inputs = []
        parsed_attributes = {}
        for i, input_ in enumerate(self.inputs):
            current_dict = {}
            annotation = (str(i), input_.type_)
            if input_ in self.nonblock_variables:
                title = input_.name
                parsed_attributes = None
            else:
                input_block = self.block_from_variable(input_)
                try:
                    block_docstring = input_block._docstring()
                    if input_ in block_docstring:
                        parsed_attributes[str(i)] = block_docstring[input_]
                except Exception:
                    parsed_attributes[(str(i))] = FAILED_ATTRIBUTE_PARSING
                if input_block.name:
                    name = input_block.name + ' - ' + input_.name
                    title = prettyname(name)
                else:
                    title = prettyname(input_.name)

            annotation_jsonschema = jsonschema_from_annotation(annotation=annotation, title=title, order=i + 1,
                                                               jsonschema_element=current_dict,
                                                               parsed_attributes=parsed_attributes)
            # Order is i+1 because of name that is at 0
            current_dict.update(annotation_jsonschema[str(i)])
            if not input_.has_default_value:
                required_inputs.append(str(i))
            else:
                dict_ = set_default_value(jsonschema_element=current_dict, key=str(i),
                                          default_value=input_.default_value)
                current_dict.update(dict_)
            properties_dict[str(i)] = current_dict[str(i)]
        properties_dict[str(len(self.inputs) + 1)] = {'type': 'string', 'title': 'WorkflowRun Name', 'editable': True,
                                                      'order': 0, "description": "Name for the resulting WorkflowRun",
                                                      'default_value': '', 'python_typing': 'builtins.str'}
        jsonschemas['run'].update({'required': required_inputs, 'method': True,
                                   'python_typing': serialize_typing(MethodType)})
        jsonschemas['start_run'] = deepcopy(jsonschemas['run'])
        jsonschemas['start_run']['required'] = []
        return jsonschemas

    def to_dict(self, use_pointers=True, memo=None, path='#'):
        if memo is None:
            memo = {}

        self.refresh_blocks_positions()
        dict_ = Block.to_dict(self)
        blocks = [b.to_dict() for b in self.blocks]
        pipes = []
        for pipe in self.pipes:
            pipes.append((self.variable_indices(pipe.input_variable),
                          self.variable_indices(pipe.output_variable)))

        dict_.update({'blocks': blocks, 'pipes': pipes, 'output': self.variable_indices(self.outputs[0]),
                      'nonblock_variables': [v.to_dict() for v in self.nonblock_variables],
                      'package_mix': self.package_mix()})

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            var_index = self.variable_indices(variable)

            if use_pointers:
                ser_value, memo = serialize_with_pointers(value, memo=memo,
                                                          path=f"{path}/imposed_variable_values/{var_index}")

            else:
                ser_value = serialize(value)
            imposed_variable_values[var_index] = ser_value

        dict_.update({'description': self.description, 'documentation': self.documentation,
                      'imposed_variable_values': imposed_variable_values})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Workflow':
        blocks = [DessiaObject.dict_to_object(d) for d in dict_['blocks']]
        if 'nonblock_variables' in dict_:
            nonblock_variables = [dict_to_object(d) for d in dict_['nonblock_variables']]
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
            iterator = zip(dict_['imposed_variables'], dict_['imposed_variable_values'])
            for variable_index, serialized_value in iterator:
                value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                variable = temp_workflow.variable_from_index(variable_index)

                imposed_variable_values[variable] = value
        elif 'imposed_variable_values' in dict_:
            # New format with a dict
            imposed_variable_values = {}
            for variable_index, serialized_value in dict_['imposed_variable_values']:
                value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                variable = temp_workflow.variable_from_index(variable_index)
                imposed_variable_values[variable] = value

        else:
            imposed_variable_values = None

        if "description" in dict_:
            # Retro-compatibility
            description = dict_["description"]
        else:
            description = ""

        if "documentation" in dict_:
            # Retro-compatibility
            documentation = dict_["documentation"]
        else:
            documentation = ""
        return cls(blocks=blocks, pipes=pipes, output=output,
                   imposed_variable_values=imposed_variable_values,
                   description=description, documentation=documentation, name=dict_["name"])

    def dict_to_arguments(self, dict_: JsonSerializable, method: str):
        dict_ = {int(k): v for k, v in dict_.items()}  # serialisation set keys as strings
        if method in self._allowed_methods:
            arguments_values = {}
            for i, input_ in enumerate(self.inputs):
                has_default = input_.has_default_value
                if not has_default or (has_default and i in dict_):
                    value = dict_[i]
                    deserialized_value = deserialize_argument(type_=input_.type_, argument=value)
                    arguments_values[i] = deserialized_value

            name_index = len(self.inputs) + 1
            if str(name_index) in dict_:
                name = dict_[str(name_index)]
            else:
                name = None
            arguments = {'input_values': arguments_values, 'name': name}
            return arguments
        raise NotImplementedError(f"Method {method} not in Workflow allowed methods")

    def method_dict(self, method_name: str = None, method_jsonschema: Any = None):
        return {}

    def variable_from_index(self, index: Union[int, Tuple[int, int, int]]):
        """
        Index elements are, in order : (Block index : int, Port side (0: input, 1: output), Port index : int)
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

    @property
    def runtime_blocks(self):
        """
        Returns blocks that are upstream for output
        """
        output_block = self.block_from_variable(self.output)
        output_upstreams = self.upstream_blocks(output_block)
        runtime_blocks = [output_block] + output_upstreams
        i = 0
        while output_upstreams and i <= len(self.blocks):
            block_upstreams = []
            for block in output_upstreams:
                block_upstreams.extend(self.upstream_blocks(block))
            output_upstreams = block_upstreams
            runtime_blocks.extend(block_upstreams)
            i += 1
        return runtime_blocks

    @property
    def export_blocks(self):
        """
        Returns block that are not upstream for output
        """
        return [b for b in self.blocks if b not in self.runtime_blocks]

    def upstream_blocks(self, block: Block) -> List[Block]:
        """
        Returns a list of given block upstream blocks
        """
        # Setting a dict here to foresee a future use. Might be unnecessary
        upstream_variables = {"available": [], "nonblock": [], "wired": []}
        input_upstreams = [self.upstream_variable(i) for i in block.inputs]
        for variable in input_upstreams:
            if variable is None:
                upstream_variables["available"].append(variable)
            elif variable in self.nonblock_variables:
                upstream_variables["nonblock"].append(variable)
            else:
                upstream_variables["wired"].append(variable)
        upstream_blocks = [self.block_from_variable(v) for v in upstream_variables["wired"]]
        return list(set(upstream_blocks))

    def get_upstream_nbv(self, variable: VariableTypes) -> VariableTypes:
        """
        If given variable has an upstream nonblock_variable, return it
        otherwise return given variable itself
        """
        if not self.nonblock_variables:
            return variable
        upstream_variable = self.upstream_variable(variable)
        if upstream_variable is not None and upstream_variable in self.nonblock_variables:
            return upstream_variable
        return variable

    def upstream_variable(self, variable: VariableTypes) -> Optional[VariableTypes]:
        """
        Returns upstream variable if given variable is connected to a pipe as a pipe output

        :param variable: Variable to search an upstream for
        """
        incoming_pipes = [p for p in self.pipes if p.output_variable == variable]
        if incoming_pipes:  # Inputs can only be connected to one pipe
            incoming_pipe = incoming_pipes[0]
            return incoming_pipe.input_variable
        return None

    def variable_indices(self, variable: VariableTypes) -> Union[Tuple[int, int, int], int]:
        """
        Returns global adress of given variable as a tuple or an int

        If variable is non block, return index of variable in variables sequence
        Else returns global adress (ib, i, ip)
        """
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
        if upstream_variable in self.nonblock_variables:
            # Free variable not attached to block
            return self.nonblock_variables.index(upstream_variable)
        raise WorkflowError(f"Something is wrong with variable {variable.name}")

    def block_from_variable(self, variable) -> Block:
        """
        Returns block of which given variable is attached to
        """
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
        warnings.warn("index method is deprecated, use input_index instead", DeprecationWarning)
        return self.input_index(variable)

    def input_index(self, variable: VariableTypes) -> Optional[int]:
        """
        If variable is a workflow input, returns its index
        """
        upstream_variable = self.get_upstream_nbv(variable)
        if upstream_variable in self.inputs:
            return self.inputs.index(upstream_variable)
        return None

    def variable_index(self, variable: VariableTypes) -> int:
        """
        Returns variable index in variables sequence
        """
        return self.variables.index(variable)

    def block_inputs_global_indices(self, block_index: int) -> List[int]:
        """
        Returns given block inputs global indices in inputs sequence
        """
        block = self.blocks[block_index]
        indices = [self.input_index(i) for i in block.inputs]
        return [i for i in indices if i is not None]

    def layout(self, min_horizontal_spacing=300, min_vertical_spacing=200, max_height=800, max_length=1500):
        """
        Computes workflow layout
        """
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

        horizontal_spacing = max(min_horizontal_spacing, max_length / max_distance)

        for i, distance in enumerate(sorted(elements_by_distance.keys())[::-1]):
            vertical_spacing = min(min_vertical_spacing, max_height / len(elements_by_distance[distance]))
            for j, element in enumerate(elements_by_distance[distance]):
                coordinates[element] = (i * horizontal_spacing, (j + 0.5) * vertical_spacing)
        return coordinates

    def refresh_blocks_positions(self):
        coordinates = self.layout()
        for i, block in enumerate(self.blocks):
            block.position = coordinates[block]
        for i, nonblock in enumerate(self.nonblock_variables):
            nonblock.position = coordinates[nonblock]

    def plot_graph(self):

        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, self.blocks, node_shape='s', node_color='grey')
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

    def run(self, input_values, verbose=False, progress_callback=lambda x: None, name=None):
        """
        Full run of a workflow. Yields a WorkflowRun
        """
        log = ''

        state = self.start_run(input_values)
        state.activate_inputs(check_all_inputs=True)

        start_time = time.time()

        log_msg = 'Starting workflow run at {}'
        log_line = log_msg.format(time.strftime('%d/%m/%Y %H:%M:%S UTC', time.gmtime(start_time)))
        log += (log_line + '\n')
        if verbose:
            print(log_line)

        state.continue_run(progress_callback=progress_callback)

        end_time = time.time()
        log_line = f"Workflow terminated in {end_time - start_time} s"

        log += log_line + '\n'
        if verbose:
            print(log_line)

        state.output_value = state.values[self.outputs[0]]

        if not name:
            name = self.name + ' run'
        return state.to_workflow_run(name=name)

    def start_run(self, input_values=None):
        """
        Partial run of a workflow. Yields a WorkflowState
        """
        return WorkflowState(self, input_values=input_values)

    def jointjs_data(self):
        coordinates = self.layout()
        blocks = []
        for block in self.blocks:
            # TOCHECK Is it necessary to add is_workflow_input/output for outputs/inputs ??
            block_data = block.jointjs_data()
            inputs = [{'name': i.name, 'is_workflow_input': i in self.inputs,
                       'has_default_value': hasattr(i, 'default_value')} for i in block.inputs]
            outputs = [{'name': o.name, 'is_workflow_output': o in self.outputs} for o in block.outputs]
            block_data.update({'inputs': inputs, 'outputs': outputs, 'position': coordinates[block]})
            blocks.append(block_data)

        nonblock_variables = []
        for variable in self.nonblock_variables:
            is_input = variable in self.inputs
            nonblock_variables.append({'name': variable.name, 'is_workflow_input': is_input,
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
        data.update({'blocks': blocks, 'nonblock_variables': nonblock_variables, 'edges': edges})
        return data

    def plot(self):
        """
        Display workflow in web browser
        """
        data = json.dumps(self.jointjs_data())
        rendered_template = workflow_template.substitute(workflow_data=data)

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
                        issubclass(pipe.input_variable.type_, pipe.output_variable.type_)
                    except TypeError:  # TODO: need of a real typing check
                        consistent = True
                        if not consistent:
                            raise TypeError(f"Inconsistent pipe type from pipe input {pipe.input_variable.name}"
                                            f"to pipe output {pipe.output_variable.name}: "
                                            f"{pipe.input_variable.type_} incompatible with"
                                            f"{pipe.output_variable.type_}")
        return True

    def package_mix(self):
        """
        Compute a structure showing percentages of packages used
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

    def __init__(self, workflow: Workflow, name: str = ''):
        self.workflow = workflow
        # TODO: configuring port internal connections
        self.input_connections = None
        self.output_connections = None
        inputs = []
        for i, variable in enumerate(self.workflow.inputs):
            input_ = variable.copy()
            input_.name = f"{name} - {variable.name}"
            inputs.append(input_)

        outputs = [self.workflow.output.copy()]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return hash(self.workflow)

    def equivalent(self, other):
        if not Block.equivalent(self, other):
            return False
        return self.workflow == other.workflow

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self)
        dict_.update({'workflow': self.workflow.to_dict()})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_):
        return cls(workflow=Workflow.dict_to_object(dict_['workflow']), name=dict_['name'])

    def evaluate(self, values):
        args = {self.inputs.index(input_): v for input_, v in values.items()}
        workflow_run = self.workflow.run(args)
        return [workflow_run.output_value]

    def package_mix(self):
        return self.workflow.package_mix()

    def _docstring(self):
        workflow_docstrings = self.workflow._docstring()
        docstring = {}
        for block_docstring in workflow_docstrings:
            for input_ in self.workflow.inputs:
                if block_docstring and input_ in block_docstring:
                    docstring[input_] = block_docstring[input_]
        return docstring


class WorkflowState(DessiaObject):
    _standalone_in_db = True
    _allowed_methods = ['block_evaluation', 'evaluate_next_block', 'continue_run',
                        'evaluate_maximum_blocks', 'add_block_input_values']
    _non_serializable_attributes = ['activated_items']

    def __init__(self, workflow: Workflow, input_values=None, activated_items=None, values=None,
                 start_time: float = None, end_time: float = None, output_value=None, log: str = '', name: str = ''):
        """
        A workflow State represents the state of execution of a workflow.
        """
        self.workflow = workflow
        if input_values is None:
            input_values = {}
        self.input_values = input_values

        if activated_items is None:
            activated_items = {p: False for p in workflow.pipes}
            activated_items.update({v: False for v in workflow.variables})
            activated_items.update({b: False for b in workflow.blocks})
        self.activated_items = activated_items

        if values is None:
            values = {}
        self.values = values

        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        self.end_time = end_time

        self.output_value = output_value
        self.log = log

        self.activate_inputs()
        DessiaObject.__init__(self, name=name)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        workflow = self.workflow.copy(deep=True, memo=memo)
        input_values = deepcopy_value(value=self.input_values, memo=memo)
        values = {}
        for variable, value in self.values.items():
            variable_indices = self.workflow.variable_indices(variable)
            copied_variable = workflow.variable_from_index(variable_indices)
            values[copied_variable] = deepcopy_value(value=value, memo=memo)

        activated_items = {}
        for item, value in self.activated_items.items():
            if isinstance(item, Variable):
                variable_indices = self.workflow.variable_indices(item)
                copied_item = workflow.variable_from_index(variable_indices)
            elif isinstance(item, Block):
                block_index = self.workflow.blocks.index(item)
                copied_item = workflow.blocks[block_index]
            elif isinstance(item, Pipe):
                pipe_index = self.workflow.pipes.index(item)
                copied_item = workflow.pipes[pipe_index]
            else:
                raise ValueError(f"WorkflowState Copy Error : item {item} cannot be activated")
            activated_items[copied_item] = value

        output_value = deepcopy_value(value=self.output_value, memo=memo)

        workflow_state = self.__class__(workflow=workflow, input_values=input_values, activated_items=activated_items,
                                        values=values, start_time=self.start_time, end_time=self.end_time,
                                        output_value=output_value, log=self.log, name=self.name)
        return workflow_state

    def _data_hash(self):
        progress = int(100 * self.progress)
        workflow = hash(self.workflow)
        output = choose_hash(self.output_value)
        input_values = sum([i * choose_hash(v) for i, v in self.input_values.items()])
        values = sum([len(k.name) * choose_hash(v) for k, v in self.values.items()])
        return (progress + workflow + output + input_values + values) % 1000000000

    def _data_eq(self, other_object: 'WorkflowState'):
        if not (self.__class__.__name__ == other_object.__class__.__name__
                and self.progress == other_object.progress
                and self.workflow == other_object.workflow
                and self.input_values.keys() == other_object.input_values.keys()
                and self.output_value == other_object.output_value):
            return False

        for index, value in self.input_values.items():
            if value != other_object.input_values[index]:
                return False

        for block, other_block in zip(self.workflow.blocks, other_object.workflow.blocks):
            if self.activated_items[block] != other_object.activated_items[other_block]:
                # Check block progress state
                return False
            variables = block.inputs + block.outputs
            other_variables = other_block.inputs + other_block.outputs
            for variable, other_variable in zip(variables, other_variables):
                if self.activated_items[variable] != other_object.activated_items[other_variable]:
                    # Check variables progress state
                    return False

                if self.activated_items[variable] and variable in self.values:
                    if self.values[variable] != other_object.values[other_variable]:
                        # Check variable values for evaluated ones
                        return False

        for pipe, other_pipe in zip(self.workflow.pipes, other_object.workflow.pipes):
            if self.activated_items[pipe] != other_object.activated_items[other_pipe]:
                # Check pipe progress state
                return False
        return True

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        """
        Transform object into a dict
        """
        # if not use_pointers:
        #     raise NotImplementedError('WorkflowState to_dict should not be called with use_pointers=False')
        if memo is None:
            memo = {}

        if use_pointers:
            workflow_dict = self.workflow.to_dict(path=f'{path}/workflow', memo=memo)
        else:
            workflow_dict = self.workflow.to_dict(use_pointers=False)

        dict_ = self.base_dict()
        dict_['workflow'] = workflow_dict

        input_values = {}
        for input_, value in self.input_values.items():
            if use_pointers:
                serialized_v, memo = serialize_with_pointers(value=value, memo=memo,
                                                             path=f"{path}/input_values/{input_}")
            else:
                serialized_v = serialize(value)
            input_values[input_] = serialized_v

        dict_['input_values'] = input_values

        # Output value: priority for reference before values
        if self.output_value is not None:
            if use_pointers:
                serialized_output_value, memo = serialize_with_pointers(self.output_value, memo=memo,
                                                                        path=f'{path}/output_value')
            else:
                serialized_output_value = serialize(self.output_value)

            dict_.update({'output_value': serialized_output_value,
                          'output_value_type': recursive_type(self.output_value)})

        # Values
        if use_pointers:
            values = {}
            for variable, value in self.values.items():
                variable_index = self.workflow.variable_index(variable)
                serialized_value, memo = serialize_with_pointers(value=value, memo=memo,
                                                                 path=f"{path}/values/{variable_index}")
                values[variable_index] = serialized_value
        else:
            values = {self.workflow.variable_index(i): serialize(v) for i, v in self.values.items()}

        dict_['values'] = values

        dict_['evaluated_blocks_indices'] = [i for i, b in enumerate(self.workflow.blocks)
                                             if b in self.activated_items and self.activated_items[b]]

        dict_['evaluated_pipes_indices'] = [i for i, p in enumerate(self.workflow.pipes)
                                            if p in self.activated_items and self.activated_items[p]]

        dict_['evaluated_variables_indices'] = [self.workflow.variable_indices(v) for v in self.workflow.variables
                                                if v in self.activated_items and self.activated_items[v]]

        dict_.update({'start_time': self.start_time, 'end_time': self.end_time, 'log': self.log})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'WorkflowState':

        # This is copy pasted from generic dict to object, because it is difficult to do a decorator
        # for both classmethod and function
        if pointers_memo is None:
            pointers_memo = {}

        if global_dict is None:
            global_dict = dict_
            pointers_memo.update(dereference_jsonpointers(dict_))

        workflow = Workflow.dict_to_object(dict_['workflow'])
        if 'output_value' in dict_:  # and 'output_value_type' in dict_:
            # type_ = dict_['output_value_type']
            value = dict_['output_value']
            output_value = deserialize(value, global_dict=dict_,
                                       pointers_memo=pointers_memo, path=f'{path}/output_value')
        else:
            output_value = None

        values = {workflow.variables[int(i)]: deserialize(v,
                                                          global_dict=dict_,
                                                          pointers_memo=pointers_memo)
                  for i, v in dict_['values'].items()}

        input_values = {int(i): deserialize(v, global_dict=dict_, pointers_memo=pointers_memo)
                        for i, v in dict_['input_values'].items()}

        activated_items = {b: (True if i in dict_['evaluated_blocks_indices'] else False)
                           for i, b in enumerate(workflow.blocks)}

        activated_items.update({p: (True if i in dict_['evaluated_pipes_indices'] else False)
                                for i, p in enumerate(workflow.pipes)})

        var_indices = []
        for variable_indices in dict_['evaluated_variables_indices']:
            if is_sequence(variable_indices):
                var_indices.append(tuple(variable_indices))  # json serialisation loses tuples
            else:
                var_indices.append(variable_indices)
        activated_items.update({v: (True if workflow.variable_indices(v) in var_indices else False)
                                for v in workflow.variables})

        return cls(workflow=workflow, input_values=input_values, activated_items=activated_items,
                   values=values, start_time=dict_['start_time'], end_time=dict_['end_time'],
                   output_value=output_value, log=dict_['log'], name=dict_['name'])

    def add_input_value(self, input_index: int, value):
        """
        Add a value for given input
        """  # TODO: Type checking?
        self.input_values[input_index] = value
        self.activate_inputs()

    def add_several_input_values(self, indices: List[int], values):
        """
        Add several values for given inputs
        """
        for index in indices:
            input_ = self.workflow.inputs[index]
            if index not in values:
                if self.activated_items[input_]:
                    value = self.values[input_]
                else:
                    msg = f"Value {input_.name} of index {index} in inputs has no value"
                    if isinstance(input_, TypedVariable):
                        msg += f": should be instance of {input_.type_}"
                    raise ValueError(msg)
            else:
                value = values[index]
            self.add_input_value(input_index=index, value=value)

    def add_block_input_values(self, block_index: int, values):
        """
        Add inputs values for given block
        """
        values = {int(k): v for k, v in values.items()}  # serialisation set keys as strings
        indices = self.workflow.block_inputs_global_indices(block_index)
        self.add_several_input_values(indices=indices, values=values)

    def _displays(self) -> List[JsonSerializable]:
        data = self.to_dict()

        display_object = DisplayObject(type_='workflow_state', data=data)
        displays = [display_object.to_dict()]
        return displays

    @property
    def progress(self):
        """
        Return the current progress, a float between 0. (nothing has been evaluated),
        to 1. (every computational block evaluated)
        """
        evaluated_blocks = [self.activated_items[b] for b in self.workflow.runtime_blocks]
        progress = sum(evaluated_blocks) / len(evaluated_blocks)
        if progress == 1 and self.end_time is None:
            self.end_time = time.time()
        return progress

    def block_evaluation(self, block_index: int, progress_callback=lambda x: None) -> bool:
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
        return None

    def continue_run(self, progress_callback=lambda x: None, export: bool = False):
        """
        Evaluate all possible blocks
        """
        self.activate_inputs()

        evaluated_blocks = []
        something_activated = True
        while something_activated and (self.progress < 1 or export):
            something_activated = False

            for pipe in self._activable_pipes():
                self._evaluate_pipe(pipe)
                something_activated = True

            for block in self._activable_blocks():
                evaluated_blocks.append(block)
                self._evaluate_block(block)
                if not export:
                    progress_callback(self.progress)
                something_activated = True
        return evaluated_blocks

    def _activable_pipes(self):
        """
        Returns all current activable pipes
        """
        pipes = []
        for i, pipe in enumerate(self.workflow.pipes):
            if not self.activated_items[pipe] and self.activated_items[pipe.input_variable]:
                pipes.append(pipe)
        return pipes

    def _activate_activable_pipes(self):
        """
        Activates current acitvable pipes
        """
        activable_pipes = self._activable_pipes()
        for pipe in activable_pipes:
            self._evaluate_pipe(pipe)

    def _activable_blocks(self):
        """
        Returns a list of all activable blocks, ie blocks that have all inputs ready for evaluation
        """
        if self.progress < 1:
            blocks = self.workflow.runtime_blocks
        else:
            blocks = self.workflow.export_blocks
        return [b for b in blocks if not self.activated_items[b] and self._block_activable_by_inputs(b)]

    def _block_activable_by_inputs(self, block: Block):
        """
        Returns wether a block has all its inputs active and can be activated
        """
        for function_input in block.inputs:
            if not self.activated_items[function_input]:
                return False
        return True

    def _evaluate_pipe(self, pipe):
        """
        Propagate data between the two variables linked by the pipe, and store it into the object
        """
        self.activated_items[pipe] = True
        self.values[pipe.output_variable] = self.values[pipe.input_variable]
        self.activated_items[pipe.output_variable] = True

    def _evaluate_block(self, block, progress_callback=lambda x: x, verbose=False):
        """
        Evaluate given block
        """
        if verbose:
            log_line = f"Evaluating block {block.name}"
            self.log += log_line + '\n'
            if verbose:
                print(log_line)

        output_values = block.evaluate({i: self.values[i] for i in block.inputs})
        # Updating progress
        if progress_callback is not None:
            progress_callback(self.progress)

        # Unpacking result of evaluation
        output_items = zip(block.outputs, output_values)
        for output, output_value in output_items:
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
                msg = f"Value {variable.name} of index {index} in inputs has no value"
                if isinstance(variable, TypedVariable):
                    msg += f": should be instance of {variable.type_}"
                raise ValueError(msg)

    def to_workflow_run(self, name=''):
        """
        If state is complete, returns a WorkflowRun
        """
        if self.progress == 1:
            values = {v: self.values[v] for v in self.workflow.variables if v.memorize and v in self.values}
            return WorkflowRun(workflow=self.workflow, input_values=self.input_values, output_value=self.output_value,
                               values=values, activated_items=self.activated_items, start_time=self.start_time,
                               end_time=self.end_time, log=self.log, name=name)
        raise ValueError('Workflow not completed')

    def _export_formats(self):
        """
        Reads block to compute available export formats
        """
        export_formats = []
        for i, block in enumerate(self.workflow.blocks):
            if hasattr(block, "_export_format"):
                export_formats.append(block._export_format(i))
        return export_formats

    def export(self, block_index: int):
        """
        Perform export
        """
        if self.progress >= 1:
            block = self.workflow.blocks[block_index]
            # TODO We should track different Export branches and run the only one concerned.
            #  Should we use evaluate_block ?
            self.continue_run(export=True)
            output = block.outputs[0]
            return self.values[output]
        raise RuntimeError("Workflow has not reached its output and cannot be exported")


class WorkflowRun(WorkflowState):
    _standalone_in_db = True
    _allowed_methods = ['run_again']
    _eq_is_data_eq = True
    _jsonschema = {
        "definitions": {}, "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
        "standalone_in_db": True, "title": "WorkflowRun Base Schema", "required": [],
        "python_typing": 'dessia_common.workflow.Pipe',
        "properties": {
            "workflow": {"type": "object", "title": "Workflow", "python_typing": "dessia_common.workflow.Workflow",
                         "classes": ["dessia_common.workflow.Workflow"], "order": 0,
                         "editable": False, "description": "Workflow"},
            'output_value': {"type": "object", "classes": "Any", "title": "Values",
                             "description": "Input and output values", "editable": False,
                             "order": 1, "python_typing": "Any"},
            'input_values': {
                'type': 'object', 'order': 2, 'editable': False,
                'title': 'Input Values', "python_typing": "Dict[str, Any]",
                'patternProperties': {
                    '.*': {'type': "object", 'classes': 'Any'}
                }
            },
            'variable_values': {
                'type': 'object', 'order': 3, 'editable': False,
                'title': 'Variables Values', "python_typing": "Dict[str, Any]",
                'patternProperties': {
                    '.*': {'type': "object", 'classes': 'Any'}
                }
            },
            'start_time': {"type": "number", "title": "Start Time", "editable": False, "python_typing": "builtins.int",
                           "description": "Start time of simulation", "order": 4},
            'end_time': {"type": "number", "title": "End Time", "editable": False, "python_typing": "builtins.int",
                         "description": "End time of simulation", "order": 5},
            'log': {"type": "string", "title": "Log", "editable": False, 'python_typing': 'builtins.str',
                    "description": "Log", "order": 6},
            "name": {'type': 'string', 'title': 'Name', 'editable': True, 'order': 7,
                     'default_value': '', 'python_typing': 'builtins.str'}
        }
    }

    def __init__(self, workflow: Workflow, input_values, output_value, values,
                 activated_items: Dict[Union[Pipe, Block, Variable], bool],
                 start_time: float, end_time: float = None, log: str = "", name: str = ""):
        if end_time is None:
            end_time = time.time()
        self.end_time = end_time
        self.execution_time = end_time - start_time
        WorkflowState.__init__(self, workflow=workflow, input_values=input_values, activated_items=activated_items,
                               values=values, start_time=start_time, output_value=output_value, log=log, name=name)

    @property
    def variable_values(self):
        """
        Jsonable saved values
        """
        return {self.workflow.variable_indices(k): v for k, v in self.values.items() if k.memorize}

    def _displays(self) -> List[JsonSerializable]:
        # Init display with workflow view
        displays = self.workflow._displays()

        # Find & order displayable blocks
        d_blocks = [b for b in self.workflow.blocks if hasattr(b, 'display_')]
        sorted_d_blocks = sorted(d_blocks, key=lambda b: b.order)

        self._activate_activable_pipes()
        self.activate_inputs()
        for block in sorted_d_blocks:
            if block in self._activable_blocks():
                self._evaluate_block(block)
            reference_path = ''
            local_values = {}
            for i, input_ in enumerate(block.inputs):
                input_adress = self.workflow.variable_indices(input_)
                strindices = str(input_adress)
                local_values[input_] = self.variable_values[input_adress]
                if i == block._displayable_input:
                    reference_path = 'variable_values/' + strindices
            display_ = block.display_(local_values=local_values, reference_path=reference_path)
            displays.extend(display_)
        if isinstance(self.output_value, DessiaObject):
            displays.extend(self.output_value._displays(reference_path='output_value'))
        return displays

    def dict_to_arguments(self, dict_: JsonSerializable, method: str):
        if method in self._allowed_methods:
            return self.workflow.dict_to_arguments(dict_=dict_, method='run')
        raise NotImplementedError(f"Method {method} not in WorkflowRun allowed methods")

    def method_dict(self, method_name: str = None, method_jsonschema: Any = None):
        if method_name is not None and method_name == 'run_again' and method_jsonschema is not None:
            dict_ = serialize_dict(self.input_values)
            for property_, value in method_jsonschema['properties'].items():
                if property_ in dict_ and 'object_id' in value and 'object_class' in value:
                    # TODO : Check. this is probably useless as we are not dealing with default values here
                    dict_[property_] = value
            return dict_
        # TODO Check this result. Might raise an error
        return DessiaObject.method_dict(self, method_name=method_name, method_jsonschema=method_jsonschema)

    def run_again(self, input_values, progress_callback=None, name=None):
        """
        Execute workflow again with given inputs
        """
        return self.workflow.run(input_values=input_values, verbose=False,
                                 progress_callback=progress_callback, name=name)

    @property
    def _method_jsonschemas(self):
        jsonschemas = self.workflow._method_jsonschemas
        jsonschemas['run_again'] = jsonschemas.pop('run')
        jsonschemas['run_again']['classes'] = ["dessia_common.workflow.WorkflowRun"]
        return jsonschemas


def set_inputs_from_function(method, inputs=None):
    """
    Inspect given method argspecs and sets block inputs from it
    """
    if inputs is None:
        inputs = []
    args_specs = inspect.getfullargspec(method)
    nargs, ndefault_args = split_argspecs(args_specs)

    for iarg, argument in enumerate(args_specs.args[1:]):
        if argument not in ['self', 'progress_callback']:
            try:
                annotations = get_type_hints(method)
                type_ = type_from_annotation(annotations[argument], module=method.__module__)
            except KeyError:
                raise UntypedArgumentError(f"Argument {argument} of method/function {method.__name__} has no typing")
            if iarg >= nargs - ndefault_args:
                default = args_specs.defaults[ndefault_args - nargs + iarg]
                input_ = TypedVariableWithDefaultValue(type_=type_, default_value=default, name=argument)
                inputs.append(input_)
            else:
                inputs.append(TypedVariable(type_=type_, name=argument))
    return inputs


def output_from_function(function, name: str = "result output"):
    """
    Inspects given function argspecs and compute block output from it
    """
    annotations = get_type_hints(function)
    if 'return' in annotations:
        type_ = type_from_annotation(annotations['return'], function.__module__)
        return TypedVariable(type_=type_, name=name)
    return Variable(name=name)


def value_type_check(value, type_):
    try:  # TODO: Subscripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass
    return True
