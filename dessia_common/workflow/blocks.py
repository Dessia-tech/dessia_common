#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:52:32 2022

@author: masfaraud
"""

import inspect
import warnings

from zipfile import ZipFile
from typing import List, Type, Any, Dict, get_type_hints

import itertools
from dessia_common.core import DessiaFilter, FiltersList, enhanced_deep_attr, split_argspecs,\
    type_from_annotation, DessiaObject
from dessia_common.utils.types import get_python_class_from_class_name, full_classname
from dessia_common.utils.docstrings import parse_docstring, EMPTY_PARSED_ATTRIBUTE
from dessia_common.utils.serialization import serialize
from dessia_common.errors import UntypedArgumentError
from dessia_common.typings import JsonSerializable, MethodType, ClassMethodType
from dessia_common.files import StringFile, BinaryFile
from dessia_common.utils.helpers import concatenate
from dessia_common.breakdown import attrmethod_getter

from dessia_common.workflow.core import Block, Variable, TypedVariable, TypedVariableWithDefaultValue,\
    set_block_variable_names_from_dict, Workflow, DisplaySetting, DisplayObject
from dessia_common.workflow.utils import ToScriptElement


def set_inputs_from_function(method, inputs=None):
    """
    Inspect given method argspecs and sets block inputs from it
    """
    if inputs is None:
        inputs = []
    args_specs = inspect.getfullargspec(method)
    nargs, ndefault_args = split_argspecs(args_specs)

    for iarg, argument in enumerate(args_specs.args):
        if argument not in ['self', 'cls', 'progress_callback']:
            try:
                annotations = get_type_hints(method)
                type_ = type_from_annotation(annotations[argument], module=method.__module__)
            except KeyError as error:
                raise UntypedArgumentError(f"Argument {argument} of method/function {method.__name__} has no typing")\
                    from error
            if iarg > nargs - ndefault_args:
                default = args_specs.defaults[ndefault_args - nargs + iarg - 1]
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


class BlockError(Exception):
    pass


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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
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

    def _to_script(self) -> ToScriptElement:
        script = f"InstantiateModel(model_class=" \
                 f"{self.model_class.__name__}, name='{self.name}')"

        imports = [
            full_classname(object_=self.model_class, compute_for='class'),
            self.full_classname
        ]
        return ToScriptElement(declaration=script, imports=imports)


class ClassMethod(Block):
    """
    Handles static method as well
    """

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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
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

    def _to_script(self) -> ToScriptElement:
        script = f"ClassMethod(method_type=ClassMethodType(" \
                 f"{self.method_type.class_.__name__}, '{self.method_type.name}')" \
                 f", name='{self.name}')"

        imports = [
            full_classname(object_=self.method_type, compute_for='instance'),
            full_classname(object_=self.method_type.class_, compute_for='class'),
            self.full_classname
        ]

        return ToScriptElement(declaration=script, imports=imports)


class ModelMethod(Block):
    """
    :param method_type: Represent class and method used.
    :type method_type: MethodType[T]
    :param name: Name of the block.
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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
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

    def _to_script(self) -> ToScriptElement:
        script = f"ModelMethod(method_type=MethodType(" \
                 f"{self.method_type.class_.__name__}, '{self.method_type.name}')" \
                 f", name='{self.name}')"

        imports = [
            full_classname(object_=self.method_type, compute_for='instance'),
            full_classname(object_=self.method_type.class_, compute_for='class'),
            self.full_classname
        ]
        return ToScriptElement(declaration=script, imports=imports)


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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['number_arguments'] = self.number_arguments
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['number_arguments'], dict_['name'])

    def evaluate(self, values):
        return [[values[var] for var in self.inputs]]

    def _to_script(self) -> ToScriptElement:
        script = f"Sequence(number_arguments={len(self.inputs)}, name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Concatenate(Block):
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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['number_arguments'] = self.number_arguments
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['number_arguments'], dict_['name'])

    def evaluate(self, values: Dict[Variable, Any]):
        list_values = list(values.values())
        return [concatenate(list_values)]

    def _to_script(self) -> ToScriptElement:
        script = f"Concatenate(number_arguments={len(self.inputs)}, name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


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
        for variable in self.workflow.inputs:
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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_.update({'workflow': self.workflow.to_dict(use_pointers=use_pointers, memo=memo, path=f'{path}/workflow')})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'WorkflowBlock':
        workflow = Workflow.dict_to_object(dict_=dict_["workflow"])
        # , force_generic = force_generic,
        # global_dict = global_dict, pointers_memo = pointers_memo,
        # path = f"{path}/workflow"
        return cls(workflow=workflow, name=dict_['name'])

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

    def _to_script(self) -> ToScriptElement:
        prefix = 'sub_'
        workflow_script = self.workflow._to_script(prefix)
        script_workflow = f"\n# --- Subworkflow --- \n" \
            f"{workflow_script.declaration}" \
            f"# --- End Subworkflow --- \n"

        script = f"WorkflowBlock(workflow={prefix}workflow, name='{self.name}')"

        imports = workflow_script.imports + [self.full_classname]
        return ToScriptElement(declaration=script, before_declaration=script_workflow, imports=imports)


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
                variable_name = 'Iterable input: ' + workflow_input.name
                inputs.append(Variable(name=variable_name))
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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        wb_dict = self.workflow_block.to_dict(use_pointers=use_pointers, memo=memo, path=f"{path}/worklow_block")
        dict_.update({'workflow_block': wb_dict, 'iter_input_index': self.iter_input_index})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        workflow_block = WorkflowBlock.dict_to_object(dict_=dict_['workflow_block'])
        # , force_generic = force_generic,
        # global_dict = global_dict, pointers_memo = pointers_memo,
        # path = f"{path}/workflow_block"
        return cls(workflow_block=workflow_block, iter_input_index=dict_['iter_input_index'], name=dict_['name'])

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
        for input_, workflow_input in zip(self.inputs, self.workflow_block.workflow.inputs):
            block_docstring[input_] = wb_docstring[workflow_input]
        return block_docstring

    def _to_script(self) -> ToScriptElement:
        wfblock_script_elements = self.workflow_block._to_script()
        wfblock_script = f"{wfblock_script_elements.before_declaration}\n" \
                         f"wfblock = {wfblock_script_elements.declaration}"
        foreach_script = f"ForEach(workflow_block=wfblock, iter_input_index={self.iter_input_index})"

        imports = wfblock_script_elements.imports + [self.full_classname]
        return ToScriptElement(declaration=foreach_script, before_declaration=wfblock_script, imports=imports)


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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['indices'] = self.indices
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Unpacker':
        return cls(dict_['indices'], dict_['name'])

    def evaluate(self, values):
        return [values[self.inputs[0]][i] for i in self.indices]

    def _to_script(self) -> ToScriptElement:
        script = f"Unpacker(indices={self.indices}, name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


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

    def _to_script(self) -> ToScriptElement:
        script = f"Flatten(name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['number_list'] = self.number_list
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

    def _to_script(self) -> ToScriptElement:
        script = f"Product(number_list={self.number_list}, name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Filter(Block):
    """
    :param filters: A list of dictionaries, each corresponding to a value to filter.
                    The dictionary should be as follows :
                    *{'attribute' : Name of attribute to filter (str),
                      'comparison_operator' : choose between gt, lt, get, let (standing for greater than, lower than,
                                                                    geater or equal than, lower or equal than) (str),
                      'bound' :  the value (float)}*
    :type filters: list[DessiaFilter]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, filters: List[DessiaFilter], logical_operator: str = "and", name: str = ''):
        self.filters = filters
        self.logical_operator = logical_operator
        inputs = [Variable(name='input_list')]
        outputs = [Variable(name='output_list')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.filters == other.filters

    def equivalent_hash(self):
        hashes = [hash(f) for f in self.filters]
        return int(sum(hashes) % 10e5)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        filters_dict = [f.to_dict(use_pointers=use_pointers, memo=memo, path=f"{path}/filters/{i}")
                        for i, f in enumerate(self.filters)]
        dict_.update({"filters": filters_dict, "logical_operator": self.logical_operator})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        filters = [DessiaFilter.dict_to_object(dict_=d, force_generic=force_generic, global_dict=global_dict,
                                               pointers_memo=pointers_memo, path=f"{path}/filters/{i}")
                   for i, d in enumerate(dict_['filters'])]
        return cls(filters=filters, logical_operator=dict_["logical_operator"], name=dict_["name"])

    def evaluate(self, values):
        filters_list = FiltersList(self.filters, self.logical_operator)
        return [filters_list.apply(values[self.inputs[0]])]

    def _to_script(self) -> ToScriptElement:
        filter_variables = [f"DessiaFilter("
                            f"attribute='{f.attribute}', comparison_operator='{f.comparison_operator}', "
                            f"bound={f.bound}, name='{f.name}')" for f in self.filters]
        filters = '[' + ",".join(filter_variables) + ']'
        script = f"Filter(filters={filters}, logical_operator='{self.logical_operator}', name='{self.name}')"

        imports = [DessiaFilter("", "", 0).full_classname, self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class Display(Block):
    _displayable_input = 0
    _non_editable_attributes = ['inputs']

    def __init__(self, inputs: List[Variable] = None, order: int = None, name: str = ''):
        """
        Abstract class for display behaviors
        """
        if order is not None:
            warnings.warn("Display Block : order argument is deprecated and will be removed in a future version."
                          "You can safely remove it from your block definition", DeprecationWarning)
        self.order = order
        if inputs is None:
            self.warn_deprecation()
            inputs = [TypedVariable(type_=DessiaObject, name='Model to Display')]
        output = TypedVariable(type_=DisplayObject, name="Display Object")
        Block.__init__(self, inputs=inputs, outputs=[output], name=name)

        self._type = None
        self._selector = None

    @staticmethod
    def warn_deprecation():
        warnings.warn("Display Block used as a generator for the displays of an object is deprecated."
                      "ts display behavior will be faulty. Please use the specific block"
                      "to generate wanted displays (MultiPlot, CadView, PlotData, Markdown)", DeprecationWarning)

    @property
    def type_(self):
        if self._type:
            return self._type
        if self.__class__ is Display:
            self.warn_deprecation()
            return ""
        raise NotImplementedError(f"type_ attribute is not implemented for block of type '{type(self)}'")

    @property
    def selector(self):
        if self._selector:
            return self._selector
        if self.__class__ is Display:
            self.warn_deprecation()
            return ""
        raise NotImplementedError(f"selector attribute is not implemented for block of type '{type(self)}'")

    def _display_settings(self, block_index: int) -> DisplaySetting:
        args = {'block_index': block_index}
        return DisplaySetting(selector=None, type_=self.type_, method="block_display", serialize_data=True,
                              arguments=args)

    def evaluate(self, values):
        object_ = values[self.inputs[0]]
        settings = object_._display_settings_from_selector(self.selector)
        data = attrmethod_getter(object_, settings.method)()
        return [DisplayObject(type_=settings.type, data=serialize(data), name=self.name)]

    def _to_script(self) -> ToScriptElement:
        script = f"{self.__class__.__name__}(name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class MultiPlot(Display):
    """
    Generates a PlotData multiplot which axes will be the given attributes
    (can be deep attributes with the '/' separator)

    :param attributes: A List of all attributes that will be shown on axes in the ParallelPlot window.
    :type attributes: List[str]
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, attributes: List[str], order: int = None, name: str = ''):
        if order is not None:
            warnings.warn("Display Block : order argument is deprecated and will be removed in a future version."
                          "You can safely remove it from your block definition", DeprecationWarning)
        self.order = order
        self.attributes = attributes
        Display.__init__(self, inputs=[TypedVariable(List[DessiaObject])], name=name)
        self.inputs[0].name = 'Input List'
        self._type = "plot_data"
        self._selector = None

    def equivalent(self, other):
        same_attributes = self.attributes == other.attributes
        return Block.equivalent(self, other) and same_attributes

    def equivalent_hash(self):
        return sum(len(a) for a in self.attributes)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['attributes'] = self.attributes
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(attributes=dict_['attributes'], name=dict_['name'])

    def evaluate(self, values):
        import plot_data
        objects = values[self.inputs[self._displayable_input]]
        attr_values = [{a: enhanced_deep_attr(o, a) for a in self.attributes} for o in objects]
        values2d = [{key: val[key]} for key in self.attributes[:2] for val in attr_values]
        tooltip = plot_data.Tooltip(name='Tooltip', attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip, x_variable=self.attributes[0], y_variable=self.attributes[1],
                                        elements=values2d, name='Scatter Plot')

        parallelplot = plot_data.ParallelPlot(disposition='horizontal', axes=self.attributes,
                                              rgbs=[(192, 11, 11), (14, 192, 11), (11, 11, 192)], elements=attr_values)
        plots = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=attr_values, plots=plots, sizes=sizes,
                                            coords=[(0, 0), (0, 300)], name='Results plot')
        return [DisplayObject(type_=self.type_, data=[multiplot.to_dict()])]

    def _to_script(self) -> ToScriptElement:
        script = f"MultiPlot(attributes={self.attributes}, name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class CadView(Display):
    """
    Generates a DisplayObject that is displayable in 3D Viewer features (BabylonJS, ...)

    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, name: str = ''):
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], name=name)

        self._type = "babylon_data"
        self._selector = "cad"


class Markdown(Display):
    """
    Generates the markdown representation of an object

    :param name: Name of the block.
    :type name: str
    """
    def __init__(self, name: str = ''):
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], name=name)

        self._type = "markdown"
        self._selector = "markdown"


class PlotData(Display):
    """
    Generates a DisplayObject that is displayable in PlotData features.
    Uses the the input object's plot_data method.

    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, name: str = ''):
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], name=name)

        self._type = "plot_data"
        self._selector = "plot_data"


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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_.update({'attribute_name': self.attribute_name})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['attribute_name'], dict_['name'])

    def evaluate(self, values):
        return [enhanced_deep_attr(values[self.inputs[0]], self.attribute_name)]

    def _to_script(self) -> ToScriptElement:
        script = f"ModelAttribute(attribute_name='{self.attribute_name}', name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class SetModelAttribute(Block):
    """
    :param attribute_name: The name of the attribute to set.
    :type attribute_name: str
    :param name: The name of the block.
    :type name: str
    """

    def __init__(self, attribute_name: str, name: str = ''):
        self.attribute_name = attribute_name
        inputs = [Variable(name='Model'), Variable(name=f'Value to insert for attribute {attribute_name}')]
        outputs = [Variable(name=f'Model with changed attribute {attribute_name}')]
        Block.__init__(self, inputs, outputs, name=name)

    def equivalent_hash(self):
        return 3 + len(self.attribute_name)

    def equivalent(self, other):
        return Block.equivalent(self, other) and self.attribute_name == other.attribute_name

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_.update({'attribute_name': self.attribute_name})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(dict_['attribute_name'], dict_['name'])

    def evaluate(self, values):
        model = values[self.inputs[0]]
        setattr(model, self.attribute_name, values[self.inputs[1]])
        return [model]

    def _to_script(self) -> ToScriptElement:
        script = f"SetModelAttribute(attribute_name='{self.attribute_name}', name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


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
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
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

    def _to_script(self) -> ToScriptElement:
        script = f"Sum(number_elements={self.number_elements}, name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Substraction(Block):
    def __init__(self, name: str = ''):
        Block.__init__(self, [Variable(name='+'), Variable(name='-')], [Variable(name='Substraction')], name=name)

    def evaluate(self, values):
        return [values[self.inputs[0]] - values[self.inputs[1]]]

    def _to_script(self) -> ToScriptElement:
        script = f"Substraction(name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class ConcatenateStrings(Block):
    """
    Concatenates the n input elements, separate by the separator input, into one string

    :param number_elements: Number of block inputs
    :type number_elements: int
    :param separator: Character used to joins the input elements together
    :type separator: str
    """
    def __init__(self, number_elements: int = 2, separator: str = "", name: str = ''):
        self.number_elements = number_elements
        self.separator = separator
        inputs = [TypedVariableWithDefaultValue(name=f"Substring {i + 1}", type_=str, default_value="")
                  for i in range(number_elements)]
        output = TypedVariable(name="Concatenation", type_=str)
        Block.__init__(self, inputs=inputs, outputs=[output], name=name)

    def equivalent_hash(self):
        return self.number_elements + hash(self.separator)

    def equivalent(self, other):
        same_number = self.number_elements == other.number_elements
        same_separator = self.separator == other.separator
        return Block.equivalent(self, other) and same_number and same_separator

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_.update({'number_elements': self.number_elements, "separator": self.separator})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        return cls(number_elements=dict_['number_elements'], separator=dict_["separator"], name=dict_['name'])

    def evaluate(self, values):
        chunks = [values[i] for i in self.inputs]
        return [self.separator.join(chunks)]

    def _to_script(self) -> ToScriptElement:
        script = f"ConcatenateStrings(number_elements={self.number_elements}, " \
                 f"separator='{self.separator}', " \
                 f"name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Export(Block):
    def __init__(self, method_type: MethodType[Type],  text: bool, extension: str,
                 filename: str = "export", name: str = ""):
        """
        Block that enables an export of an object calling its configured method.
        Only Methods that yields streams (and not files) should be used.

        The file generated will be called {filename}.{extension}

        :param method_type: An object that have a class_ input (which is the class of the incoming model)
            and a name (which is the name of the method that will be called).
        :type method_type: MethodType[T]
        :param text: Whether the export is of type text or not
        :type text: bool
        :param extension: Extension of the resulting file (ex: json or xlsx)
        :type extension: str
        :param filename: Name of the resulting file without its extension
        :type filename: str
        :param name: Name of the block.
        :type name: str
        """
        self.method_type = method_type
        if not filename:
            filename = "export"
        self.filename = filename

        method = method_type.get_method()

        self.extension = extension
        self.text = text

        output = output_from_function(function=method, name="export_output")
        inputs = [TypedVariable(type_=method_type.class_, name="model_to_export"),
                  TypedVariableWithDefaultValue(type_=str, default_value=filename, name="filename")]
        Block.__init__(self, inputs=inputs, outputs=[output], name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        classname = full_classname(object_=self.method_type.class_, compute_for='class')
        method_type_dict = {'class_': classname, 'name': self.method_type.name}
        dict_.update({"method_type": method_type_dict, "extension": self.extension,
                      "text": self.text, "filename": self.filename})
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Export':
        class_ = get_python_class_from_class_name(dict_['method_type']['class_'])
        method_type = MethodType(class_=class_, name=dict_['method_type']['name'])
        if "export_name" in dict_:
            # RetroCompat
            filename = dict_["export_name"]
        else:
            filename = dict_["filename"]
        return cls(method_type=method_type, text=dict_['text'], filename=filename,
                   extension=dict_["extension"], name=dict_["name"])

    def evaluate(self, values):
        filename = f"{values.pop(self.inputs[-1])}.{self.extension}"
        if self.text:
            stream = StringFile(filename)
        else:
            stream = BinaryFile(filename)
        getattr(values[self.inputs[0]], self.method_type.name)(stream)
        return [stream]

    def _export_format(self, block_index: int):
        args = {"block_index": block_index}
        return {"extension": self.extension, "method_name": "export", "text": self.text,
                "export_name": self.filename, "args": args}

    def _to_script(self) -> ToScriptElement:
        script = f"Export(method_type=MethodType(" \
                 f"{full_classname(object_=self.method_type.class_, compute_for='class')}, '{self.method_type.name}')" \
                 f", filename='{self.filename}', extension='{self.extension}'" \
                 f", text={self.text}, name='{self.name}')"

        imports = [self.full_classname, full_classname(object_=self.method_type, compute_for='instance'),
                   full_classname(object_=self.method_type.class_, compute_for='class')]
        return ToScriptElement(declaration=script, imports=imports)


class Archive(Block):
    def __init__(self, number_exports: int = 1, filename: str = "archive", name: str = ""):
        """
        A block that takes n inputs and store them in a archive (ZIP,...)

        :param number_exports: The number of files that will be stored in the archive
        :type number_exports: int
        :param filename: Name of the resulting archive file without its extension
        :type filename: str
        :param name: Name of the block.
        :type name: str
        """
        self.number_exports = number_exports
        self.filename = filename
        inputs = [Variable(name="export_" + str(i)) for i in range(number_exports)]
        inputs.append(TypedVariableWithDefaultValue(type_=str, default_value=filename, name="filename"))
        Block.__init__(self, inputs=inputs, outputs=[Variable(name="zip archive")], name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['number_exports'] = len(self.inputs) - 1   # Filename is also a block input
        dict_["filename"] = self.filename
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        if "export_name" in dict_:
            # RetroCompat
            filename = dict_["export_name"]
        else:
            filename = dict_["filename"]
        return cls(number_exports=dict_["number_exports"], filename=filename, name=dict_['name'])

    def evaluate(self, values):
        name_input = self.inputs[-1]
        archive_name = f"{values.pop(name_input)}.zip"
        archive = BinaryFile(archive_name)
        with ZipFile(archive, 'w') as zip_archive:
            for input_ in self.inputs[:-1]:  # Filename is last block input
                value = values[input_]
                if isinstance(value, StringFile):
                    with zip_archive.open(value.filename, 'w') as file:
                        file.write(value.getvalue().encode('utf-8'))
                elif isinstance(value, BinaryFile):
                    with zip_archive.open(value.filename, 'w') as file:
                        file.write(value.getbuffer())
                else:
                    raise ValueError(f"Archive input is not a file-like object. Got '{value}' of type {type(value)}")
        return [archive]

    def _export_format(self, block_index: int):
        return {"extension": "zip", "method_name": "export", "text": False,
                "export_name": self.filename, "args": {"block_index": block_index}}

    def _to_script(self) -> ToScriptElement:
        script = f"Archive(number_exports={self.number_exports}, filename='{self.filename}', name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])
