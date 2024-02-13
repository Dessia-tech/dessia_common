#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module to define Blocks for workflows. """

import inspect
import warnings
from zipfile import ZipFile
from typing import List, Type, Any, Dict, Tuple, get_type_hints, TypeVar, Optional
import itertools
from dessia_common.core import DessiaFilter, FiltersList, type_from_annotation, DessiaObject
from dessia_common.schemas.core import split_argspecs, parse_docstring, EMPTY_PARSED_ATTRIBUTE
from dessia_common.displays import DisplaySetting, DisplayObject
from dessia_common.errors import UntypedArgumentError
from dessia_common.typings import (JsonSerializable, MethodType, ClassMethodType, AttributeType, ViewType, CadViewType,
                                   PlotDataType, MarkdownType)
from dessia_common.files import StringFile, BinaryFile, generate_archive
from dessia_common.utils.helpers import concatenate, full_classname, get_python_class_from_class_name
from dessia_common.breakdown import attrmethod_getter, get_in_object_from_path
from dessia_common.exports import ExportFormat
from dessia_common.workflow.core import Block, Variable, Workflow
from dessia_common.workflow.utils import ToScriptElement

T = TypeVar("T")

Position = Tuple[float, float]


def set_inputs_from_function(method, inputs=None):
    """ Inspect given method argspecs and sets block inputs from it. """
    if inputs is None:
        inputs = []
    args_specs = inspect.getfullargspec(method)
    nargs, ndefault_args = split_argspecs(args_specs)

    for iarg, argument in enumerate(args_specs.args):
        if argument not in ["self", "cls", "progress_callback"]:
            try:
                annotations = get_type_hints(method)
                type_ = type_from_annotation(annotations[argument], module=method.__module__)
            except KeyError as error:
                raise UntypedArgumentError(f"Argument {argument} of method/function {method.__name__} has no typing")\
                    from error
            if iarg > nargs - ndefault_args:
                default = args_specs.defaults[ndefault_args - nargs + iarg - 1]
                inputs.append(Variable(type_=type_, default_value=default, name=argument))
            else:
                inputs.append(Variable(type_=type_, name=argument))
    return inputs


def output_from_function(function, name: str = "result output"):
    """ Inspect given function argspecs and compute block output from it. """
    annotations = get_type_hints(function)
    if "return" in annotations:
        type_ = type_from_annotation(annotations['return'], function.__module__)
        return Variable(type_=type_, name=name)
    return Variable(name=name)


def set_block_variable_names_from_dict(func):
    """ Inspect function arguments to compute black variable names. """
    def func_wrapper(cls, dict_):
        obj = func(cls, dict_)
        if "input_names" in dict_:
            for input_name, input_ in zip(dict_["input_names"], obj.inputs):
                input_.name = input_name
        if "output_names" in dict_:
            output_items = zip(dict_["output_names"], obj.outputs)
            for output_name, output_ in output_items:
                output_.name = output_name
        return obj
    return func_wrapper


class BlockError(Exception):
    """ Specific BlockError Exception. """


class InstantiateModel(Block):
    """
    Instantiate given class during workflow execution.

    :param model_class: Class to instantiate.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, model_class: Type, name: str = "Instantiate Model", position:  Position = (0, 0)):
        self.model_class = model_class
        inputs = []
        inputs = set_inputs_from_function(self.model_class.__init__, inputs)
        outputs = [Variable(type_=self.model_class, name="Model")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return len(self.model_class.__name__)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        classname = self.model_class.__class__.__name__
        other_classname = other.model_class.__class__.__name__
        return super().equivalent(other) and classname == other_classname

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'InstantiateModel':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        model_class = get_python_class_from_class_name(dict_["model_class"])
        block = cls(model_class=model_class, name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Instantiate a model of given class with arguments that are in values. """
        arguments = {var.name: values[var] for var in self.inputs}
        return [self.model_class(**arguments)]

    def package_mix(self):
        """ Add block contribution to workflow's package_mix. """
        return {self.model_class.__module__.split(".")[0]: 1}

    def _docstring(self):
        """ Parse given class' docstring. """
        docstring = self.model_class.__doc__
        annotations = get_type_hints(self.model_class.__init__)
        parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        parsed_attributes = parsed_docstring["attributes"]
        block_docstring = {i: parsed_attributes[i.name] if i.name in parsed_attributes
                           else EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"InstantiateModel(model_class=" \
                 f"{self.model_class.__name__}, {self.base_script()})"
        imports = [full_classname(object_=self.model_class, compute_for="class"), self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class ClassMethod(Block):
    """
    Run given class method during workflow execution. Handle static method as well.

    :param method_type: Denotes the class and method to run.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    _non_serializable_attributes = ["method"]

    def __init__(self, method_type: ClassMethodType[Type], name: str = "Class Method", position:  Position = (0, 0)):
        self.method_type = method_type
        inputs = []

        self.method = method_type.get_method()
        inputs = set_inputs_from_function(self.method, inputs)

        self.argument_names = [i.name for i in inputs]

        output = output_from_function(function=self.method, name="Return")
        super().__init__(inputs, [output], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        classname = self.method_type.class_.__name__
        return len(classname) + 7 * len(self.method_type.name)

    def equivalent(self, other: 'ClassMethod'):
        """ Return whether the block is equivalent to the other given or not. """
        classname = self.method_type.class_.__name__
        other_classname = other.method_type.class_.__name__
        same_class = classname == other_classname
        same_method = self.method_type.name == other.method_type.name
        return super().equivalent(other) and same_class and same_method

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'ClassMethod':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        # Backward compatibility dessia_common < 0.14.0
        if "object_class" not in dict_["method_type"]:
            dict_["method_type"]["object_class"] = "dessia_common.typings.ClassMethodType"

        method_type = ClassMethodType.dict_to_object(dict_["method_type"])
        block = cls(method_type=method_type, name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Run given classmethod with arguments that are in values. """
        arguments = {arg_name: values[var] for arg_name, var in zip(self.argument_names, self.inputs) if var in values}
        return [self.method(**arguments)]

    def _docstring(self):
        """ Parse given method's docstring. """
        docstring = self.method.__doc__
        annotations = get_type_hints(self.method)
        parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        parsed_attributes = parsed_docstring["attributes"]
        block_docstring = {i: parsed_attributes[i.name] if i.name in parsed_attributes
                           else EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"ClassMethod(method_type=ClassMethodType(" \
                 f"{self.method_type.class_.__name__}, '{self.method_type.name}')" \
                 f", {self.base_script()})"

        imports = [full_classname(object_=self.method_type, compute_for="instance"),
                   full_classname(object_=self.method_type.class_, compute_for="class"),
                   self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class ModelMethod(Block):
    """
    Run given method during workflow execution.

    :param method_type: Denotes the class and method to run.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    _non_serializable_attributes = ["method"]

    def __init__(self, method_type: MethodType[Type], name: str = "Model Method", position:  Position = (0, 0)):
        self.method_type = method_type
        inputs = [Variable(type_=method_type.class_, name="Model")]
        self.method = method_type.get_method()
        inputs = set_inputs_from_function(self.method, inputs)

        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        return_output = output_from_function(function=self.method, name="Return")
        model_output = Variable(type_=method_type.class_, name="Model")
        outputs = [return_output, model_output]

        if name == "":
            name = f"Model method: {method_type.name}"
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        classname = self.method_type.class_.__name__
        return len(classname) + 7 * len(self.method_type.name)

    def equivalent(self, other: 'ModelMethod'):
        """ Return whether the block is equivalent to the other given or not. """
        classname = self.method_type.class_.__name__
        other_classname = other.method_type.class_.__name__
        same_model = classname == other_classname
        same_method = self.method_type.name == other.method_type.name
        return super().equivalent(other) and same_model and same_method

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'ModelMethod':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        method_type = MethodType.dict_to_object(dict_["method_type"])
        block = cls(method_type=method_type, name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, progress_callback=lambda x: None, **kwargs):
        """ Run given method with arguments that are in values. """
        arguments = {n: values[v] for n, v in zip(self.argument_names, self.inputs[1:]) if v in values}
        method = getattr(values[self.inputs[0]], self.method_type.name)
        try:
            # Trying to inject progress callback to method
            result = method(progress_callback=progress_callback, **arguments)
        except TypeError:
            result = method(**arguments)

        return [result, values[self.inputs[0]]]

    def package_mix(self):
        """ Add block contribution to workflow's package_mix. """
        return {self.method_type.class_.__module__.split(".")[0]: 1}

    def _docstring(self):
        """ Parse given method's docstring. """
        docstring = self.method.__doc__
        annotations = get_type_hints(self.method)
        parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        parsed_attributes = parsed_docstring["attributes"]
        block_docstring = {i: parsed_attributes[i.name] if i.name in parsed_attributes
                           else EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"ModelMethod(method_type=MethodType(" \
                 f"{self.method_type.class_.__name__}, '{self.method_type.name}')" \
                 f", {self.base_script()})"

        imports = [full_classname(object_=self.method_type, compute_for="instance"),
                   full_classname(object_=self.method_type.class_, compute_for="class"),
                   self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class Sequence(Block):
    """
    Concatenate n inputs into a sequence.

    :param number_arguments: Number of inputs to be concatenated.
    :param name: Block name.
    :param position: Position in canvas.
    """

    def __init__(self, number_arguments: int, name: str = "Sequence", position:  Position = (0, 0)):
        self.number_arguments = number_arguments
        inputs = [Variable(name=f"Sequence element {i}") for i in range(self.number_arguments)]
        outputs = [Variable(type_=List[T], name="Sequence")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_arguments

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.number_arguments == other.number_arguments

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Sequence':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(number_arguments=dict_["number_arguments"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Pack values into a sequence. """
        return [[values[var] for var in self.inputs]]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Sequence(number_arguments={len(self.inputs)}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Concatenate(Block):
    """
    Concatenate the n inputs.

    :param number_arguments: Number of input to, concatenate.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, number_arguments: int = 2, name: str = "Concatenate", position:  Position = (0, 0)):
        self.number_arguments = number_arguments
        inputs = [Variable(name=f"Sequence element {i}") for i in range(self.number_arguments)]
        outputs = [Variable(type_=List[T], name="Sequence")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_arguments

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.number_arguments == other.number_arguments

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Concatenate':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(number_arguments=dict_["number_arguments"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values: Dict[Variable, Any], **kwargs):
        """ Concatenate elements that are in values. """
        list_values = list(values.values())
        return [concatenate(list_values)]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Concatenate(number_arguments={len(self.inputs)}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class WorkflowBlock(Block):
    """
    Wrapper around workflow to put it in a block of another workflow.

    Even if a workflow is a block, it cannot be used directly as it has a different behavior
    than a Block in eq and hash which is problematic to handle in dictionaries for example.

    :param workflow: The WorkflowBlock's workflow
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, workflow: Workflow, name: str = "Workflow Block", position:  Position = (0, 0)):
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
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return hash(self.workflow)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        if not super().equivalent(other):
            return False
        return self.workflow == other.workflow

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'WorkflowBlock':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        workflow = Workflow.dict_to_object(dict_["workflow"])
        block = cls(workflow=workflow, name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Format sub workflow arguments and run it. """
        arguments = {self.inputs.index(input_): v for input_, v in values.items()}
        workflow_run = self.workflow.run(arguments)
        return [workflow_run.output_value]

    def package_mix(self):
        """ Recursively add block contribution to workflow's package_mix. """
        return self.workflow.package_mix()

    def _docstring(self):
        """ Recursively get docstring of sub workflow. """
        workflow_docstrings = self.workflow._docstring()
        docstring = {}
        for block_docstring in workflow_docstrings:
            for input_ in self.workflow.inputs:
                if block_docstring and input_ in block_docstring:
                    docstring[input_] = block_docstring[input_]
        return docstring

    def _to_script(self, prefix: str) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        prefix = f"{prefix}sub_"
        workflow_script = self.workflow._to_script(prefix)
        script_workflow = f"\n# --- Subworkflow --- \n" \
            f"{workflow_script.declaration}" \
            f"# --- End Subworkflow --- \n"

        script = f"WorkflowBlock(workflow={prefix}workflow, {self.base_script()})"

        imports = workflow_script.imports + [self.full_classname]
        return ToScriptElement(declaration=script, before_declaration=script_workflow, imports=imports)


class ForEach(Block):
    """
    A block to iterate on an input and perform an parallel for (iterations are not dependent).

    :param workflow_block: The WorkflowBlock on which iterate.
    :param iter_input_index: Index of iterable input in worklow_block.inputs
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, workflow_block: 'WorkflowBlock', iter_input_index: int, name: str = "For Each",
                 position:  Position = (0, 0)):
        self.workflow_block = workflow_block
        self.iter_input_index = iter_input_index
        self.iter_input = self.workflow_block.inputs[iter_input_index]
        inputs = []
        for i, workflow_input in enumerate(self.workflow_block.inputs):
            if i == iter_input_index:
                variable_name = f"Iterable input: {workflow_input.name}"
                inputs.append(Variable(name=variable_name))
            else:
                input_ = workflow_input.copy()
                input_.name = f"Binding: {input_.name}"
                inputs.append(input_)
        output_variable = Variable(name="Foreach output")
        self.output_connections = None  # TODO: configuring port internal connections
        self.input_connections = None
        super().__init__(inputs, [output_variable], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        wb_hash = int(self.workflow_block.equivalent_hash() % 10e5)
        return wb_hash + self.iter_input_index

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        input_eq = self.iter_input_index == other.iter_input_index
        wb_eq = self.workflow_block.equivalent(other.workflow_block)
        return super().equivalent(other) and wb_eq and input_eq

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'ForEach':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        workflow_block = WorkflowBlock.dict_to_object(dict_["workflow_block"])
        block = cls(workflow_block=workflow_block, iter_input_index=dict_["iter_input_index"],
                    name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Loop on input list and run sub workflow on each. """
        values_workflow = {var2: values[var1] for var1, var2 in zip(self.inputs, self.workflow_block.inputs)}
        output_values = []
        for value in values_workflow[self.iter_input]:
            values_workflow[self.iter_input] = value
            output = self.workflow_block.evaluate(values_workflow)[0]
            output_values.append(output)
        return [output_values]

    def _docstring(self):
        """ Recursively get docstring of sub-workflow. """
        wb_docstring = self.workflow_block._docstring()
        block_docstring = {}
        for input_, workflow_input in zip(self.inputs, self.workflow_block.workflow.inputs):
            block_docstring[input_] = wb_docstring[workflow_input]
        return block_docstring

    def _to_script(self, prefix: str) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        wfblock_script_elements = self.workflow_block._to_script(prefix)
        wfblock_script = f"{wfblock_script_elements.before_declaration}\n" \
                         f"wfblock = {wfblock_script_elements.declaration}"
        foreach_script = f"ForEach(workflow_block=wfblock, iter_input_index={self.iter_input_index}, " \
                         f"{self.base_script()})"

        imports = wfblock_script_elements.imports + [self.full_classname]
        return ToScriptElement(declaration=foreach_script, before_declaration=wfblock_script, imports=imports)


class Unpacker(Block):
    """
    DeMUX block.

    :param indices: List of indices to extract.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, indices: List[int], name: str = "Unpacker", position:  Position = (0, 0)):
        self.indices = indices
        outputs = [Variable(name=f"Element {i}") for i in indices]
        super().__init__(inputs=[Variable(name="Sequence")], outputs=outputs, name=name, position=position)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.indices == other.indices

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Unpacker':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(indices=dict_["indices"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return len(self.indices)

    def evaluate(self, values, **kwargs):
        """ Unpack input list elements into n outputs. """
        return [values[self.inputs[0]][i] for i in self.indices]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Unpacker(indices={self.indices}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Flatten(Block):
    """
    A block to extract the first element of a list and flatten it.

    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, name: str = "Flatten", position:  Position = (0, 0)):
        inputs = [Variable(name="Sequence")]
        outputs = [Variable(name="Flattened sequence")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return 1

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Flatten':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Extract the first element of a list and flatten it. """
        output = []
        for value in values[self.inputs[0]]:
            output.extend(value)
        return [output]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Flatten({self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Product(Block):
    """ A block to generate the product combinations. """

    def __init__(self, number_list: int, name: str = "Product", position:  Position = (0, 0)):
        self.number_list = number_list
        inputs = [Variable(name=f"Sequence {i}") for i in range(self.number_list)]
        output_variable = Variable(name="Product")
        super().__init__(inputs, [output_variable], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_list

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.number_list == other.number_list

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Product':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(number_list=dict_["number_list"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Compute the block: use itertools.product. """
        list_product = [values[var] for var in self.inputs]
        output_value = list(itertools.product(*list_product))
        return [output_value]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Product(number_list={self.number_list}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Filter(Block):
    """
    A Block to filter some data according to some filters.

    :param filters: A list of DessiaFilters, each corresponding to a value to filter.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, filters: List[DessiaFilter], logical_operator: str = "and", name: str = "Filter",
                 position:  Position = (0, 0)):
        self.filters = filters
        self.logical_operator = logical_operator
        inputs = [Variable(name="Sequence")]
        outputs = [Variable(name="Filtered sequence")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.filters == other.filters

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Filter':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        filters = [DessiaFilter.dict_to_object(f) for f in dict_["filters"]]
        block = cls(filters=filters, logical_operator=dict_["logical_operator"],
                    name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        hashes = [hash(f) for f in self.filters]
        return int(sum(hashes) % 10e5)

    def evaluate(self, values, **kwargs):
        """ Apply given filters to input list. """
        filters_list = FiltersList(self.filters, self.logical_operator)
        return [filters_list.apply(values[self.inputs[0]])]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        filter_variables = [f"DessiaFilter("
                            f"attribute='{f.attribute}', comparison_operator='{f.comparison_operator}', "
                            f"bound={f.bound}, name='{f.name}')" for f in self.filters]
        filters = f"[{','.join(filter_variables)}"
        script = f"Filter(filters={filters}, logical_operator='{self.logical_operator}', {self.base_script()})"

        imports = [DessiaFilter("", "", 0).full_classname, self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class Display(Block):
    """ Abstract block class for display behaviors. """

    _displayable_input = 0
    _non_editable_attributes = ["inputs"]
    _type = (0, 0)
    serialize = False

    def __init__(self, inputs: List[Variable], load_by_default: bool = False, name: str = "Display",
                 selector: Optional[ViewType] = None, position:  Position = (0, 0)):
        output = Variable(type_=DisplayObject, name="Display Object")
        super().__init__(inputs=inputs, outputs=[output], name=name, position=position)

        self.load_by_default = load_by_default
        self.selector = selector

    @property
    def type_(self) -> str:
        """ Get display's type_. """
        if self._type:
            return self._type
        raise NotImplementedError(f"type_ attribute is not implemented for block of type '{type(self)}'")

    def _display_settings(self, block_index: int, reference_path: str = "#") -> DisplaySetting:
        """ Compute block's display settings. """
        arguments = {"block_index": block_index, "reference_path": reference_path}
        return DisplaySetting(selector=self.selector.name, type_=self.type_, method="block_display",
                              serialize_data=self.serialize, arguments=arguments,
                              load_by_default=self.load_by_default)

    def evaluate(self, values, **kwargs):
        """ Run method defined by selector's display_setting and compute corresponding DisplayObject. """
        object_ = values[self.inputs[0]]
        settings = object_._display_settings_from_selector(self.selector.name)
        method = settings.method
        if "progress_callback" in kwargs:
            # User methods do not necessarily implement progress callback
            del kwargs["progress_callback"]
        try:
            return [attrmethod_getter(object_, method)(**kwargs)]
        except TypeError as exc:
            arguments = list(kwargs.keys())
            warnings.warn(f"Workflow : method '{method}' was called without generic arguments "
                          f"('{', '.join(arguments)}') because one of them is not set in method's signature.\n\n "
                          f"Original exception : \n{repr(exc)}")
            # Cover cases where kwargs do not correspond to method signature (missing reference_path, for ex)
            return [attrmethod_getter(object_, method)()]

    @property
    def display_class_name(self):
        """ Get the class name of the current instance. """
        return self.__class__.__name__

    @property
    def selector_class_name(self):
        """ Get the class name of the Selector. """
        return self.selector.__class__.__name__

    def to_script_selector(self) -> ToScriptElement:
        """ Write Block's Selector config into a chunk of script. """
        script = f"{self.selector_class_name}(name='{self.name}')"
        selector_class_ = full_classname(self.selector.class_, compute_for="class")
        selector_class_name = f"{self.selector.__module__}.{self.selector_class_name}"
        return ToScriptElement(declaration=script, imports=[selector_class_, selector_class_name])

    def base_script(self) -> str:
        """ Generate a chunk of script that denotes the arguments of a base block. """
        return f"name=\"{self.name}\", load_by_default={self.load_by_default}, position={self.position}"

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        imports = [self.full_classname]
        script_selector = self.to_script_selector()
        script = f"{self.display_class_name}(selector={self.selector_class_name}(" \
                 f"class_={self.selector.class_.__name__}, name='{self.selector.name}')" \
                 f", {self.base_script()})"

        imports.extend(script_selector.imports)
        return ToScriptElement(declaration=script, imports=imports)


class DeprecatedMultiPlot(Display):
    """
    Generate a Multi plot which axes will be the given attributes.

    :param attributes: A List of all attributes that will be shown on axes in the ParallelPlot window.
        Can be deep attributes with the '/' separator.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    type_ = "plot_data"

    def __init__(self, attributes: List[str], load_by_default: bool = True,
                 name: str = "", position:  Position = (0, 0)):
        self.attributes = attributes
        Display.__init__(self, inputs=[Variable(type_=List[DessiaObject])], load_by_default=load_by_default,
                         name=name, position=position)
        self.inputs[0].name = "Input List"
        self.serialize = True

    def equivalent(self, other):
        """ Return whether if the block is equivalent to the other given. """
        same_attributes = self.attributes == other.attributes
        return super().equivalent(other) and same_attributes

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return sum(len(a) for a in self.attributes)

    def evaluate(self, values, **kwargs):
        """ Create MultiPlot from block configuration. Handle reference path. """
        reference_path = kwargs.get("reference_path", "#")
        import plot_data
        objects = values[self.inputs[self._displayable_input]]
        samples = [plot_data.Sample(values={a: get_in_object_from_path(o, a) for a in self.attributes},
                                    reference_path=f"{reference_path}/{i}", name=f"Sample {i}")
                   for i, o in enumerate(objects)]
        samples2d = [plot_data.Sample(values={a: get_in_object_from_path(o, a) for a in self.attributes[:2]},
                                      reference_path=f"{reference_path}/{i}", name=f"Sample {i}")
                     for i, o in enumerate(objects)]
        tooltip = plot_data.Tooltip(name="Tooltip", attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip, x_variable=self.attributes[0], y_variable=self.attributes[1],
                                        elements=samples2d, name="Scatter Plot")

        parallelplot = plot_data.ParallelPlot(disposition="horizontal", axes=self.attributes,
                                              rgbs=[(192, 11, 11), (14, 192, 11), (11, 11, 192)], elements=samples)
        plots = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=samples, plots=plots, sizes=sizes,
                                            coords=[(0, 0), (0, 300)], name="Results plot")
        return [multiplot.to_dict()]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"MultiPlot(attributes={self.attributes}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class MultiPlot(Display):
    """
    Generate a Multiplot which axes will be the given attributes. Will show a Scatter and a Parallel Plot.

    :param selector_name: Name of the selector to be displayed in object page. Must be unique throughout workflow.
    :param attributes: A List of all attributes that will be shown on axes in the ParallelPlot window.
        Can be deep attributes with the '/' separator.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    _type = "plot_data"
    serialize = True

    def __init__(self, selector_name: str, attributes: List[str], load_by_default: bool = True,
                 name: str = "Multiplot", position:  Position = (0, 0)):
        self.attributes = attributes
        Display.__init__(self, inputs=[Variable(type_=List[DessiaObject])], load_by_default=load_by_default,
                         name=name, selector=PlotDataType(class_=DessiaObject, name=selector_name), position=position)
        self.inputs[0].name = "Sequence"

    def __deepcopy__(self, memo=None):
        return MultiPlot(selector_name=self.selector.name, attributes=self.attributes,
                         load_by_default=self.load_by_default, name=self.name, position=self.position)

    def equivalent(self, other):
        """ Return whether if the block is equivalent to the other given. """
        same_attributes = self.attributes == other.attributes
        return super().equivalent(other) and same_attributes

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return sum(len(a) for a in self.attributes)

    def evaluate(self, values, **kwargs):
        """ Create MultiPlot from block configuration. Handle reference path. """
        reference_path = kwargs.get("reference_path", "#")
        import plot_data
        objects = values[self.inputs[self._displayable_input]]
        samples = [plot_data.Sample(values={a: get_in_object_from_path(o, a) for a in self.attributes},
                                    reference_path=f"{reference_path}/{i}", name=f"Sample {i}")
                   for i, o in enumerate(objects)]
        samples2d = [plot_data.Sample(values={a: get_in_object_from_path(o, a) for a in self.attributes[:2]},
                                      reference_path=f"{reference_path}/{i}", name=f"Sample {i}")
                     for i, o in enumerate(objects)]
        tooltip = plot_data.Tooltip(name="Tooltip", attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip, x_variable=self.attributes[0], y_variable=self.attributes[1],
                                        elements=samples2d, name="Scatter Plot")

        parallelplot = plot_data.ParallelPlot(disposition="horizontal", axes=self.attributes,
                                              rgbs=[(192, 11, 11), (14, 192, 11), (11, 11, 192)], elements=samples)
        plots = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=samples, plots=plots, sizes=sizes,
                                            coords=[(0, 0), (0, 300)], name="Results plot")
        return [multiplot.to_dict()]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"MultiPlot(selector_name='{self.selector.name}', attributes={self.attributes}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#',
                id_method=True, id_memo=None, **kwargs) -> JsonSerializable:
        """ Overwrite to_dict method in order to handle difference of behaviors about selector. """
        dict_ = super().to_dict(use_pointers=use_pointers, memo=memo, path=path, id_method=id_method, id_memo=id_memo)
        dict_.update({"selector_name": self.selector.name, "attributes": self.attributes, "name": self.name,
                      "load_by_default": self.load_by_default, "position": self.position})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'MultiPlot':
        """ Backward compatibility for old versions of Display blocks. """
        selector_name = dict_.get("selector_name", None)
        selector = dict_.get("selector", None)
        if selector is None and selector_name is None:
            # Backward compatibility < 0.14.0
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedMultiPlot(attributes=dict_["attributes"], name=dict_["name"],
                                       load_by_default=load_by_default, position=dict_["position"])
        if selector is not None and selector_name is None:
            if isinstance(selector, str):
                selector_name = selector
            else:
                # Backward compatibility 0.14.0 < v < 0.14.1
                selector_name = selector["name"]
        block = MultiPlot(selector_name=selector_name, attributes=dict_["attributes"], name=dict_["name"],
                          load_by_default=dict_["load_by_default"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block


class DeprecatedCadView(Display):
    """
    Deprecated version of CadView block.

    Steps to upgrade to new version (CadView) :
    - Remove type_ argument from your __init__ call
    - Argument 'selector' is now the first in the call and doesn't have default value anymore.
    - Argument 'selector' is of type CadViewType instead of str
    """

    _type = "babylon_data"

    def __init__(self, name: str = "", load_by_default: bool = False, selector: str = "cad",
                 position:  Position = (0, 0)):
        warnings.warn("This version of CadView Block is deprecated and should not be used anymore."
                      "Please upgrade to CadView new version, instead. (see docstrings)", DeprecationWarning)
        input_ = Variable(type_=DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)


class CadView(Display):
    """ Generate a DisplayObject that is displayable in 3D Viewer features (BabylonJS, ...). """

    _type = "babylon_data"

    def __init__(self, selector: CadViewType[Type], name: str = "Cad View", load_by_default: bool = False,
                 position:  Position = (0, 0)):
        if isinstance(selector, str):
            raise TypeError("Argument 'selector' should be of type 'CadViewType' and not 'str',"
                            " which is deprecated. See upgrading guide if needed.")
        input_ = Variable(type_=selector.class_, name="Model")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'CadView':
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "cad")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedCadView(name=dict_["name"], load_by_default=load_by_default, selector=selector,
                                     position=dict_["position"])
        selector = CadViewType.dict_to_object(selector)
        block = CadView(selector=selector, name=dict_["name"], load_by_default=dict_["load_by_default"],
                        position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block


class DeprecatedMarkdown(Display):
    """
    Deprecated version of Markdown block.

    Steps to upgrade to new version (Markdown) :
    - Remove type_ argument from your __init__ call
    - Argument 'selector' is now the first in the call and doesn't have default value anymore.
    - Argument 'selector' is of type MarkdownSelector instead of str.
    """

    _type = "markdown"

    def __init__(self, name: str = "", load_by_default: bool = False, selector: str = "markdown",
                 position:  Position = (0, 0)):
        warnings.warn("This version of 'Markdown' Block is deprecated and should not be used anymore."
                      "Please upgrade to 'Markdown' new version, instead. (see docstrings)", DeprecationWarning)
        input_ = Variable(type_=DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, name=name,
                         selector=selector, position=position)


class Markdown(Display):
    """ Generate a Markdown representation of an object. """

    _type = "markdown"

    def __init__(self, selector: MarkdownType[Type], name: str = "Markdown", load_by_default: bool = False,
                 position:  Position = (0, 0)):
        if isinstance(selector, str):
            raise TypeError("Argument 'selector' should be of type 'MarkdownType' and not 'str',"
                            " which is deprecated. See upgrading guide if needed.")
        input_ = Variable(type_=selector.class_, name="Model")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Markdown':
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "markdown")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedMarkdown(name=dict_["name"], load_by_default=load_by_default, selector=selector,
                                      position=dict_["position"])
        selector = MarkdownType.dict_to_object(selector)
        block = Markdown(selector=selector, name=dict_["name"], load_by_default=dict_["load_by_default"],
                         position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block


class DeprecatedPlotData(Display):
    """
    Deprecated version of PlotData block.

    Steps to upgrade to new version (PlotData) :
    - Remove type_ argument from your __init__ call
    - Argument 'selector' is now the first in the call and doesn't have default value anymore.
    - Argument 'selector' is of type PlotDataSelector instead of str.
    """

    _type = "plot_data"
    serialize = True

    def __init__(self, name: str = "", load_by_default: bool = False, selector: str = "plot_data",
                 position:  Position = (0, 0)):
        warnings.warn("This version of 'PlotData' Block is deprecated and should not be used anymore."
                      "Please upgrade to 'PlotData' new version, instead. (see docstrings)", DeprecationWarning)
        input_ = Variable(type_=DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, name=name,
                         selector=selector, position=position)


class PlotData(Display):
    """ Generate a PlotData representation of an object. """

    _type = "plot_data"
    serialize = True

    def __init__(self, selector: PlotDataType[Type], name: str = "Plot Data", load_by_default: bool = False,
                 position:  Position = (0, 0)):
        if isinstance(selector, str):
            raise TypeError("Argument 'selector' should be of type 'PlotDataType' and not 'str',"
                            " which is deprecated. See upgrading guide if needed.")
        input_ = Variable(type_=selector.class_, name="Model")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'PlotData':
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "plot_data")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedPlotData(name=dict_["name"], load_by_default=load_by_default, selector=selector,
                                      position=dict_["position"])
        selector = PlotDataType.dict_to_object(selector)
        block = PlotData(selector=selector, name=dict_["name"], load_by_default=dict_["load_by_default"],
                         position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block


class ModelAttribute(Block):
    """
    Fetch attribute of given object during workflow execution.

    :param attribute_name: The name of the attribute to select.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, attribute_name: str, name: str = "Model Attribute", position:  Position = (0, 0)):
        self.attribute_name = attribute_name
        inputs = [Variable(name="Model")]
        outputs = [Variable(name="Attribute value")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return len(self.attribute_name)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.attribute_name == other.attribute_name

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'ModelAttribute':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(attribute_name=dict_["attribute_name"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Get input object's deep attribute. """
        return [get_in_object_from_path(values[self.inputs[0]], f"#/{self.attribute_name}")]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"ModelAttribute(attribute_name='{self.attribute_name}', {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])

   
class GetModelAttribute(Block):
    """
    Fetch attribute of given object during workflow execution.

    :param attribute_type: AttributeType variable that contain the model and the name of the attribute to select.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, attribute_type: AttributeType[Type], name: str = "Get Attribute", position:  Position = (0, 0)):
        self.attribute_type = attribute_type
        parameters = inspect.signature(self.attribute_type.class_).parameters
        inputs = [Variable(type_=self.attribute_type.class_, name="Model")]
        type_ = get_attribute_type(self.attribute_type.name, parameters)
        outputs = [Variable(type_=type_, name="Attribute")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        classname = self.attribute_type.class_.__name__
        return len(classname) + 7 * len(self.attribute_type.name)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        classname = self.attribute_type.class_.__name__
        other_classname = other.attribute_type.class_.__name__
        same_model = classname == other_classname
        same_method = self.attribute_type.name == other.attribute_type.name
        return super().equivalent(other) and same_model and same_method

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'GetModelAttribute':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        attribute_type = AttributeType.dict_to_object(dict_["attribute_type"])
        block = cls(attribute_type=attribute_type, name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Get input object's deep attribute. """
        return [get_in_object_from_path(values[self.inputs[0]], f"#/{self.attribute_type.name}")]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"GetModelAttribute(attribute_type=AttributeType(" \
                 f"{self.attribute_type.class_.__name__}, name=\"{self.attribute_type.name}\")" \
                 f", {self.base_script()})"
        imports = [full_classname(object_=self.attribute_type, compute_for="instance"),
                   full_classname(object_=self.attribute_type.class_, compute_for="class"),
                   self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class SetModelAttribute(Block):
    """
    Block to set an attribute value in a workflow.

    :param attribute_type: AttributeType variable that contain the model and the name of the attribute to select.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, attribute_type: AttributeType[Type], name: str = "Set Attribute", position:  Position = (0, 0)):
        self.attribute_type = attribute_type
        parameters = inspect.signature(self.attribute_type.class_).parameters
        inputs = [Variable(type_=self.attribute_type.class_, name="Model")]
        type_ = get_attribute_type(self.attribute_type.name, parameters)
        inputs.append(Variable(type_=type_, name="Value"))
        outputs = [Variable(type_=self.attribute_type.class_, name="Model")]
        super().__init__(inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return 3 + len(self.attribute_type.name)

    def equivalent(self, other):
        """ Returns whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.attribute_type.name == other.attribute_type.name

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'SetModelAttribute':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        attribute_type = AttributeType.dict_to_object(dict_["attribute_type"])
        block = cls(attribute_type=attribute_type, name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Set input object's deep attribute with input value. """
        model = values[self.inputs[0]]
        setattr(model, self.attribute_type.name, values[self.inputs[1]])
        return [model]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"SetModelAttribute(attribute_type=AttributeType(" \
                 f"{self.attribute_type.class_.__name__}, name=\"{self.attribute_type.name}\")" \
                 f", {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Sum(Block):
    """
    Sum the n inputs.

    :param number_elements: Number of element to sum
    :param name: Name of the block
    :param position: Position of the block in the workflow
    """

    def __init__(self, number_elements: int = 2, name: str = "Sum", position:  Position = (0, 0)):
        self.number_elements = number_elements
        inputs = [Variable(name=f"Sum element {i + 1}") for i in range(number_elements)]
        super().__init__(inputs=inputs, outputs=[Variable(name="Sum")], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_elements

    def equivalent(self, other):
        """ Returns whether the block is equivalent to the other given or not. """
        return super().equivalent(other) and self.number_elements == other.number_elements

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Sum':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(number_elements=dict_["number_elements"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """
        Sum input values.

        TODO : This cannot work, we are summing a dictionary
        """
        return [sum(values)]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Sum(number_elements={self.number_elements}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Substraction(Block):
    """ Block that subtract input values. First is +, second is -. """

    def __init__(self, name: str = "Substraction", position:  Position = (0, 0)):
        super().__init__([Variable(name="+"), Variable(name="-")], [Variable(name="Substraction")], name=name,
                       position=position)

    def evaluate(self, values, **kwargs):
        """ Subtract input values. """
        return [values[self.inputs[0]] - values[self.inputs[1]]]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Substraction({self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Substraction':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block


class ConcatenateStrings(Block):
    """
    Concatenate the n input elements, separate by the separator input, into one string.

    :param number_elements: Number of block inputs
    :param separator: Character used to joins the input elements together
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, number_elements: int = 2, separator: str = "", name: str = "Concatenate Strings",
                 position:  Position = (0, 0)):
        self.number_elements = number_elements
        self.separator = separator
        inputs = [Variable(name=f"Substring {i + 1}", type_=str, default_value="") for i in range(number_elements)]
        output = Variable(name="Concatenation", type_=str)
        super().__init__(inputs=inputs, outputs=[output], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_elements + hash(self.separator)

    def equivalent(self, other):
        """ Returns whether the block is equivalent to the other given or not. """
        same_number = self.number_elements == other.number_elements
        same_separator = self.separator == other.separator
        return super().equivalent(other) and same_number and same_separator

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'ConcatenateStrings':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        block = cls(number_elements=dict_["number_elements"], separator=dict_["separator"],
                    name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Concatenate input strings with configured separator. """
        chunks = [values[i] for i in self.inputs]
        return [self.separator.join(chunks)]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"ConcatenateStrings(number_elements={self.number_elements}, " \
                 f"separator='{self.separator}', " \
                 f"name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class Export(Block):
    """
    Block that enables an export of an object calling its configured method.

    Only Methods that yields streams (and not files) should be used.
    The file generated will be called {filename}.{extension}

    :param method_type: An object that have a class_ input (which is the class of the incoming model)
        and a name (which is the name of the method that will be called).
    :param text: Whether the export is of type text or not
    :param extension: Extension of the resulting file (ex: json or xlsx)
    :param filename: Name of the resulting file without its extension
    :param name: Name of the block.
    """

    def __init__(self, method_type: MethodType[Type], text: bool, extension: str,
                 filename: str = "export", name: str = "Export", position:  Position = (0, 0)):
        self.method_type = method_type
        if not filename:
            filename = "export"
        self.filename = filename
        method = method_type.get_method()
        self.extension = extension
        self.text = text

        output = output_from_function(function=method, name="Stream")
        inputs = [Variable(type_=method_type.class_, name="Model"),
                  Variable(type_=str, default_value=filename, name="Filename")]
        super().__init__(inputs=inputs, outputs=[output], name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Export':
        """ Override base dict_to_object in order to force custom inputs from workflow builder. """
        method_type = ClassMethodType.dict_to_object(dict_["method_type"])
        block = cls(method_type=method_type, text=dict_["text"], extension=dict_["extension"],
                    filename=dict_["filename"], name=dict_["name"], position=dict_["position"])
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Generate to-be-exported stream from corresponding method. """
        filename = f"{values.pop(self.inputs[-1])}.{self.extension}"
        if self.text:
            stream = StringFile(filename)
        else:
            stream = BinaryFile(filename)
        getattr(values[self.inputs[0]], self.method_type.name)(stream)
        return [stream]

    def _export_format(self, block_index: int) -> ExportFormat:
        """ Compute block's export format. """
        arguments = {"block_index": block_index}
        return ExportFormat(selector=None, extension=self.extension, method_name="export", text=self.text,
                            export_name=self.filename, args=arguments)

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Export(method_type=MethodType(" \
                 f"{self.method_type.class_.__name__}, '{self.method_type.name}')" \
                 f", filename='{self.filename}', extension='{self.extension}'" \
                 f", text={self.text}, {self.base_script()})"

        imports = [self.full_classname, full_classname(object_=self.method_type, compute_for="instance"),
                   full_classname(object_=self.method_type.class_, compute_for="class")]
        return ToScriptElement(declaration=script, imports=imports)


class Archive(Block):
    """
    A block that takes n inputs and store them in a archive (ZIP,...).

    :param number_exports: The number of files that will be stored in the archive
    :param filename: Name of the resulting archive file without its extension
    :param name: Name of the block.
    """

    def __init__(self, number_exports: int = 1, filename: str = "archive", name: str = "Archive",
                 position:  Position = (0, 0)):
        self.number_exports = number_exports
        self.filename = filename
        self.extension = "zip"
        self.text = False
        inputs = [Variable(name=f"Export {i}") for i in range(number_exports)]
        inputs.append(Variable(type_=str, default_value=filename, name="Filename"))
        super().__init__(inputs=inputs, outputs=[Variable(name="Archive")], name=name, position=position)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = "#", id_method=True, id_memo=None,
                **kwargs):
        """ Serialize the block with custom logic. """
        dict_ = super().to_dict(use_pointers=use_pointers, memo=memo, path=path)
        dict_["number_exports"] = len(self.inputs) - 1   # Filename is also a block input
        dict_["filename"] = self.filename
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs):
        """ Custom dict_to_object method. """
        block = cls(number_exports=dict_["number_exports"], filename=dict_["filename"],
                    name=dict_["name"], position=dict_.get("position"))
        block.dict_to_inputs(dict_)
        return block

    def evaluate(self, values, **kwargs):
        """ Generate archive stream for input streams. """
        name_input = self.inputs[-1]
        archive_name = f"{values.pop(name_input)}.{self.extension}"
        archive = BinaryFile(archive_name)
        with ZipFile(archive, "w") as zip_archive:
            for input_ in self.inputs[:-1]:  # Filename is last block input
                value = values[input_]
                generate_archive(zip_archive, value)
        return [archive]

    def _export_format(self, block_index: int) -> ExportFormat:
        """ Compute block's export formats. """
        arguments = {"block_index": block_index}
        return ExportFormat(selector=None, extension=self.extension, method_name="export", text=self.text,
                            export_name=self.filename, args=arguments)

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Archive(number_exports={self.number_exports}, filename='{self.filename}', {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


def get_attribute_type(attribute_name: str, parameters):
    """ Get type of attribute name of class."""
    parameter = parameters.get(attribute_name)
    if not parameter:
        return None
    if not hasattr(parameter, "annotation"):
        return parameter
    if parameter.annotation == inspect.Parameter.empty:
        return None 
    return parameter.annotation
