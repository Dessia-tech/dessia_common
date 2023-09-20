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
from dessia_common.utils.helpers import concatenate, full_classname
from dessia_common.breakdown import attrmethod_getter, get_in_object_from_path
from dessia_common.exports import ExportFormat
from dessia_common.workflow.core import Block, Variable, TypedVariable, TypedVariableWithDefaultValue, Workflow
from dessia_common.workflow.utils import ToScriptElement

T = TypeVar("T")


def set_inputs_from_function(method, inputs=None):
    """ Inspect given method argspecs and sets block inputs from it. """
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
    """ Inspect given function argspecs and compute block output from it. """
    annotations = get_type_hints(function)
    if 'return' in annotations:
        type_ = type_from_annotation(annotations['return'], function.__module__)
        return TypedVariable(type_=type_, name=name)
    return Variable(name=name)


def set_block_variable_names_from_dict(func):
    """ Inspect function arguments to compute black variable names. """
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


class BlockError(Exception):
    """ Specific BlockError Exception. """


class InstantiateModel(Block):
    """
    Instantiate given class during workflow execution.

    :param model_class: Class to instantiate.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, model_class: Type, name: str = '', position: Tuple[float, float] = None):
        self.model_class = model_class
        inputs = []
        inputs = set_inputs_from_function(self.model_class.__init__, inputs)
        outputs = [TypedVariable(type_=self.model_class, name='Instanciated object')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return len(self.model_class.__name__)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        classname = self.model_class.__class__.__name__
        other_classname = other.model_class.__class__.__name__
        return Block.equivalent(self, other) and classname == other_classname

    def evaluate(self, values, **kwargs):
        """ Instantiate a model of given class with arguments that are in values. """
        arguments = {var.name: values[var] for var in self.inputs}
        return [self.model_class(**arguments)]

    def package_mix(self):
        """ Add block contribution to workflow's package_mix. """
        return {self.model_class.__module__.split('.')[0]: 1}

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
        imports = [full_classname(object_=self.model_class, compute_for='class'), self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class ClassMethod(Block):
    """
    Run given class method during workflow execution. Handle static method as well.

    :param method_type: Denotes the class and method to run.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    _non_serializable_attributes = ["method"]

    def __init__(self, method_type: ClassMethodType[Type], name: str = '', position: Tuple[float, float] = None):
        self.method_type = method_type
        inputs = []

        self.method = method_type.get_method()
        inputs = set_inputs_from_function(self.method, inputs)

        self.argument_names = [i.name for i in inputs]
        output_name = f"method result of {method_type.name}"
        output = output_from_function(function=self.method, name=output_name)
        Block.__init__(self, inputs, [output], name=name, position=position)

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
        return Block.equivalent(self, other) and same_class and same_method

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

        imports = [full_classname(object_=self.method_type, compute_for='instance'),
                   full_classname(object_=self.method_type.class_, compute_for='class'),
                   self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of blocks. """
        # Backward compatibility dessia_common < 0.14.0
        if "object_class" not in dict_["method_type"]:
            dict_["method_type"]["object_class"] = "dessia_common.typings.ClassMethodType"
        return super().dict_to_object(dict_=dict_, force_generic=True, global_dict=global_dict,
                                      pointers_memo=pointers_memo, path=path)


class ModelMethod(Block):
    """
    Run given method during workflow execution.

    :param method_type: Denotes the class and method to run.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    _non_serializable_attributes = ["method"]

    def __init__(self, method_type: MethodType[Type], name: str = '', position: Tuple[float, float] = None):
        self.method_type = method_type
        inputs = [TypedVariable(type_=method_type.class_, name='model at input')]
        self.method = method_type.get_method()
        inputs = set_inputs_from_function(self.method, inputs)

        # Storing argument names
        self.argument_names = [i.name for i in inputs[1:]]

        return_output_name = f"method result of {method_type.name}"
        return_output = output_from_function(function=self.method, name=return_output_name)

        model_output_name = f"model at output {method_type.name}"
        model_output = TypedVariable(type_=method_type.class_, name=model_output_name)
        outputs = [return_output, model_output]
        if name == "":
            name = f"Model method: {method_type.name}"
        Block.__init__(self, inputs, outputs, name=name, position=position)

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
        return Block.equivalent(self, other) and same_model and same_method

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
        return {self.method_type.class_.__module__.split('.')[0]: 1}

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

        imports = [full_classname(object_=self.method_type, compute_for='instance'),
                   full_classname(object_=self.method_type.class_, compute_for='class'),
                   self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of blocks. """
        # Backward compatibility dessia_common < 0.14.0
        if "object_class" not in dict_["method_type"]:
            dict_["method_type"]["object_class"] = "dessia_common.typings.MethodType"
        return super().dict_to_object(dict_=dict_, force_generic=True, global_dict=global_dict,
                                      pointers_memo=pointers_memo, path=path)


class Sequence(Block):
    """
    Concatenate n inputs into a sequence.

    :param number_arguments: Number of inputs to be concatenated.
    :param name: Block name.
    :param position: Position in canvas.
    """

    def __init__(self, number_arguments: int, name: str = '', position: Tuple[float, float] = None):
        self.number_arguments = number_arguments
        inputs = [Variable(name=f"Sequence element {i}") for i in range(self.number_arguments)]
        outputs = [TypedVariable(type_=List[T], name='sequence')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_arguments

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.number_arguments == other.number_arguments

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

    def __init__(self, number_arguments: int = 2, name: str = '', position: Tuple[float, float] = None):
        self.number_arguments = number_arguments
        inputs = [Variable(name=f"Sequence element {i}") for i in range(self.number_arguments)]
        outputs = [TypedVariable(type_=List[T], name='sequence')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_arguments

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.number_arguments == other.number_arguments

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

    Even if a workflow is a block, it can't be used directly as it has a different behavior
    than a Block in eq and hash which is problematic to handle in dictionaries for example.

    :param workflow: The WorkflowBlock's workflow
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, workflow: Workflow, name: str = '', position: Tuple[float, float] = None):
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
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return hash(self.workflow)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        if not Block.equivalent(self, other):
            return False
        return self.workflow == other.workflow

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
        prefix = f'{prefix}sub_'
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

    def __init__(self, workflow_block: 'WorkflowBlock', iter_input_index: int, name: str = '',
                 position: Tuple[float, float] = None):
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
        Block.__init__(self, inputs, [output_variable], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        wb_hash = int(self.workflow_block.equivalent_hash() % 10e5)
        return wb_hash + self.iter_input_index

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        input_eq = self.iter_input_index == other.iter_input_index
        wb_eq = self.workflow_block.equivalent(other.workflow_block)
        return Block.equivalent(self, other) and wb_eq and input_eq

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

    def __init__(self, indices: List[int], name: str = '', position: Tuple[float, float] = None):
        self.indices = indices
        outputs = [Variable(name=f"output_{i}") for i in indices]
        Block.__init__(self, inputs=[Variable(name="input_sequence")], outputs=outputs, name=name, position=position)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.indices == other.indices

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

    def __init__(self, name: str = '', position: Tuple[float, float] = None):
        inputs = [Variable(name='input_sequence')]
        outputs = [Variable(name='flatten_sequence')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return 1

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

    def __init__(self, number_list: int, name: str = '', position: Tuple[float, float] = None):
        self.number_list = number_list
        inputs = [Variable(name='list_product_' + str(i)) for i in range(self.number_list)]
        output_variable = Variable(name='Product output')
        Block.__init__(self, inputs, [output_variable], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_list

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.number_list == other.number_list

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

    def __init__(self, filters: List[DessiaFilter], logical_operator: str = "and", name: str = '',
                 position: Tuple[float, float] = None):
        self.filters = filters
        self.logical_operator = logical_operator
        inputs = [Variable(name='input_list')]
        outputs = [Variable(name='output_list')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.filters == other.filters

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
        filters = '[' + ",".join(filter_variables) + ']'
        script = f"Filter(filters={filters}, logical_operator='{self.logical_operator}', {self.base_script()})"

        imports = [DessiaFilter("", "", 0).full_classname, self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)


class Display(Block):
    """ Abstract block class for display behaviors. """

    _displayable_input = 0
    _non_editable_attributes = ['inputs']
    _type = None
    serialize = False

    def __init__(self, inputs: List[Variable], load_by_default: bool = False, name: str = "",
                 selector: Optional[ViewType] = None, position: Tuple[float, float] = None):
        output = TypedVariable(type_=DisplayObject, name="Display Object")
        Block.__init__(self, inputs=inputs, outputs=[output], name=name, position=position)

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
        return [attrmethod_getter(object_, settings.method)()]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"{self.__class__.__name__}(name='{self.name}')"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


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
                 name: str = "", position: Tuple[float, float] = None):
        self.attributes = attributes
        Display.__init__(self, inputs=[TypedVariable(List[DessiaObject])], load_by_default=load_by_default,
                         name=name, position=position)
        self.inputs[0].name = "Input List"
        self.serialize = True

    def equivalent(self, other):
        """ Return whether if the block is equivalent to the other given. """
        same_attributes = self.attributes == other.attributes
        return Block.equivalent(self, other) and same_attributes

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
        tooltip = plot_data.Tooltip(name='Tooltip', attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip, x_variable=self.attributes[0], y_variable=self.attributes[1],
                                        elements=samples2d, name='Scatter Plot')

        parallelplot = plot_data.ParallelPlot(disposition='horizontal', axes=self.attributes,
                                              rgbs=[(192, 11, 11), (14, 192, 11), (11, 11, 192)], elements=samples)
        plots = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=samples, plots=plots, sizes=sizes,
                                            coords=[(0, 0), (0, 300)], name='Results plot')
        return [multiplot.to_dict()]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"MultiPlot(attributes={self.attributes}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class MultiPlot(Display):
    """
    Generate a Multi plot which axes will be the given attributes.

    :param attributes: A List of all attributes that will be shown on axes in the ParallelPlot window.
        Can be deep attributes with the '/' separator.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    _type = "plot_data"
    serialize = True

    def __init__(self, selector: PlotDataType[Type], attributes: List[str], load_by_default: bool = True,
                 name: str = "", position: Tuple[float, float] = None):
        self.attributes = attributes
        Display.__init__(self, inputs=[TypedVariable(List[DessiaObject])], load_by_default=load_by_default,
                         name=name, selector=selector, position=position)
        self.inputs[0].name = "Input List"

    def equivalent(self, other):
        """ Return whether if the block is equivalent to the other given. """
        same_attributes = self.attributes == other.attributes
        return Block.equivalent(self, other) and same_attributes

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
        tooltip = plot_data.Tooltip(name='Tooltip', attributes=self.attributes)

        scatterplot = plot_data.Scatter(tooltip=tooltip, x_variable=self.attributes[0], y_variable=self.attributes[1],
                                        elements=samples2d, name='Scatter Plot')

        parallelplot = plot_data.ParallelPlot(disposition='horizontal', axes=self.attributes,
                                              rgbs=[(192, 11, 11), (14, 192, 11), (11, 11, 192)], elements=samples)
        plots = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300), plot_data.Window(width=560, height=300)]
        multiplot = plot_data.MultiplePlots(elements=samples, plots=plots, sizes=sizes,
                                            coords=[(0, 0), (0, 300)], name='Results plot')
        return [multiplot.to_dict()]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"MultiPlot(attributes={self.attributes}, {self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "Multiplot")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedMultiPlot(attributes=dict_["attributes"], name=dict_["name"],
                                       load_by_default=load_by_default, position=dict_["position"])
        selector = PlotDataType.dict_to_object(selector)
        return MultiPlot(selector=selector, attributes=dict_["attributes"], name=dict_["name"],
                         load_by_default=dict_["load_by_default"], position=dict_["position"])


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
                 position: Tuple[float, float] = None):
        warnings.warn("This version of CadView Block is deprecated and should not be used anymore."
                      "Please upgrade to CadView new version, instead. (see docstrings)", DeprecationWarning)
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)


class CadView(Display):
    """ Generate a DisplayObject that is displayable in 3D Viewer features (BabylonJS, ...). """

    _type = "babylon_data"

    def __init__(self, selector: CadViewType[Type], name: str = "", load_by_default: bool = False,
                 position: Tuple[float, float] = None):
        if isinstance(selector, str):
            raise TypeError("Argument 'selector' should be of type 'CadViewType' and not 'str',"
                            " which is deprecated. See upgrading guide if needed.")
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "cad")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedCadView(name=dict_["name"], load_by_default=load_by_default, selector=selector,
                                     position=dict_["position"])
        selector = CadViewType.dict_to_object(selector)
        return CadView(selector=selector, name=dict_["name"], load_by_default=dict_["load_by_default"],
                       position=dict_["position"])


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
                 position: Tuple[float, float] = None):
        warnings.warn("This version of 'Markdown' Block is deprecated and should not be used anymore."
                      "Please upgrade to 'Markdown' new version, instead. (see docstrings)", DeprecationWarning)
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, name=name,
                         selector=selector, position=position)


class Markdown(Display):
    """ Generate a Markdown representation of an object. """

    _type = "markdown"

    def __init__(self, selector: MarkdownType[Type], name: str = "", load_by_default: bool = False,
                 position: Tuple[float, float] = None):
        if isinstance(selector, str):
            raise TypeError("Argument 'selector' should be of type 'MarkdownType' and not 'str',"
                            " which is deprecated. See upgrading guide if needed.")
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "markdown")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedMarkdown(name=dict_["name"], load_by_default=load_by_default, selector=selector,
                                      position=dict_["position"])
        selector = MarkdownType.dict_to_object(selector)
        return Markdown(selector=selector, name=dict_["name"], load_by_default=dict_["load_by_default"],
                        position=dict_["position"])


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

    def __init__(self, name: str = '', load_by_default: bool = False, selector: str = "plot_data",
                 position: Tuple[float, float] = None):
        warnings.warn("This version of 'PlotData' Block is deprecated and should not be used anymore."
                      "Please upgrade to 'PlotData' new version, instead. (see docstrings)", DeprecationWarning)
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, name=name,
                         selector=selector, position=position)


class PlotData(Display):
    """ Generate a PlotData representation of an object. """

    _type = "plot_data"
    serialize = True

    def __init__(self, selector: PlotDataType[Type], name: str = "", load_by_default: bool = False,
                 position: Tuple[float, float] = None):
        if isinstance(selector, str):
            raise TypeError("Argument 'selector' should be of type 'PlotDataType' and not 'str',"
                            " which is deprecated. See upgrading guide if needed.")
        input_ = TypedVariable(DessiaObject, name="Model to display")
        Display.__init__(self, inputs=[input_], load_by_default=load_by_default, selector=selector,
                         name=name, position=position)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of Display blocks. """
        selector = dict_.get("selector", "plot_data")
        if isinstance(selector, str):
            load_by_default = dict_.get("load_by_default", False)
            return DeprecatedPlotData(name=dict_["name"], load_by_default=load_by_default, selector=selector,
                                      position=dict_["position"])
        selector = PlotDataType.dict_to_object(selector)
        return PlotData(selector=selector, name=dict_["name"], load_by_default=dict_["load_by_default"],
                        position=dict_["position"])


class ModelAttribute(Block):
    """
    Fetch attribute of given object during workflow execution.

    :param attribute_name: The name of the attribute to select.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, attribute_name: str, name: str = '', position: Tuple[float, float] = None):
        self.attribute_name = attribute_name
        inputs = [Variable(name='Model')]
        outputs = [Variable(name='Model attribute')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return len(self.attribute_name)

    def equivalent(self, other):
        """ Return whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.attribute_name == other.attribute_name

    def evaluate(self, values, **kwargs):
        """ Get input object's deep attribute. """
        return [get_in_object_from_path(values[self.inputs[0]], f'#/{self.attribute_name}')]

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

    def __init__(self, attribute_type: AttributeType[Type], name: str = '', position: Tuple[float, float] = None):
        self.attribute_type = attribute_type
        parameters = inspect.signature(self.attribute_type.class_).parameters
        inputs = [TypedVariable(type_=self.attribute_type.class_, name='Model')]
        type_ = get_attribute_type(self.attribute_type.name, parameters)
        if type_:
            outputs = [TypedVariable(type_=type_, name='Model attribute')]  
        else:
            outputs = [Variable(name='Model attribute')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

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
        return Block.equivalent(self, other) and same_model and same_method

    def evaluate(self, values, **kwargs):
        """ Get input object's deep attribute. """
        return [get_in_object_from_path(values[self.inputs[0]], f'#/{self.attribute_type.name}')]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"GetModelAttribute(attribute_type=AttributeType(" \
                 f"{self.attribute_type.class_.__name__}, name=\"{self.attribute_type.name}\")" \
                 f", {self.base_script()})"
        imports = [full_classname(object_=self.attribute_type, compute_for='instance'),
                   full_classname(object_=self.attribute_type.class_, compute_for='class'),
                   self.full_classname]
        return ToScriptElement(declaration=script, imports=imports)

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of blocks. """
        # Backward compatibility dessia_common < 0.14.0
        if "object_class" not in dict_["attribute_type"]:
            dict_["attribute_type"]["object_class"] = "dessia_common.typings.AttributeType"
        return super().dict_to_object(dict_=dict_, force_generic=True, global_dict=global_dict,
                                      pointers_memo=pointers_memo, path=path)


class SetModelAttribute(Block):
    """
    Block to set an attribute value in a workflow.

    :param attribute_type: AttributeType variable that contain the model and the name of the attribute to select.
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, attribute_type: AttributeType[Type], name: str = '', position: Tuple[float, float] = None):
        self.attribute_type = attribute_type
        parameters = inspect.signature(self.attribute_type.class_).parameters
        type_ = get_attribute_type(self.attribute_type.name, parameters)
        inputs = [TypedVariable(type_=self.attribute_type.class_, name='Model')]
        if type_:
            inputs.append(TypedVariable(type_=type_, name=f'Value to insert for attribute {self.attribute_type.name}'))
        else:
            inputs.append(Variable(name=f'Value to insert for attribute {self.attribute_type.name}'))
        outputs = [TypedVariable(type_=self.attribute_type.class_,
                                 name=f'Model with changed attribute {self.attribute_type.name}')]
        Block.__init__(self, inputs, outputs, name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return 3 + len(self.attribute_type.name)

    def equivalent(self, other):
        """ Returns whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.attribute_type.name == other.attribute_type.name

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

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Backward compatibility for old versions of blocks. """
        # Backward compatibility dessia_common < 0.14.0
        if "object_class" not in dict_["attribute_type"]:
            dict_["attribute_type"]["object_class"] = "dessia_common.typings.AttributeType"
        return super().dict_to_object(dict_=dict_, force_generic=True, global_dict=global_dict,
                                      pointers_memo=pointers_memo, path=path)


class Sum(Block):
    """
    Sum the n inputs.

    :param number_elements: Number of element to sum
    :param name: Name of the block
    :param position: Position of the block in the workflow
    """

    def __init__(self, number_elements: int = 2, name: str = '', position: Tuple[float, float] = None):
        self.number_elements = number_elements
        inputs = [Variable(name=f"Sum element {i + 1}") for i in range(number_elements)]
        Block.__init__(self, inputs=inputs, outputs=[Variable(name='Sum')], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_elements

    def equivalent(self, other):
        """ Returns whether the block is equivalent to the other given or not. """
        return Block.equivalent(self, other) and self.number_elements == other.number_elements

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

    def __init__(self, name: str = '', position: Tuple[float, float] = None):
        Block.__init__(self, [Variable(name='+'), Variable(name='-')], [Variable(name='Substraction')], name=name,
                       position=position)

    def evaluate(self, values, **kwargs):
        """ Subtract input values. """
        return [values[self.inputs[0]] - values[self.inputs[1]]]

    def _to_script(self, _) -> ToScriptElement:
        """ Write block config into a chunk of script. """
        script = f"Substraction({self.base_script()})"
        return ToScriptElement(declaration=script, imports=[self.full_classname])


class ConcatenateStrings(Block):
    """
    Concatenate the n input elements, separate by the separator input, into one string.

    :param number_elements: Number of block inputs
    :param separator: Character used to joins the input elements together
    :param name: Name of the block.
    :param position: Position of the block in canvas.
    """

    def __init__(self, number_elements: int = 2, separator: str = "", name: str = '',
                 position: Tuple[float, float] = None):
        self.number_elements = number_elements
        self.separator = separator
        inputs = [TypedVariableWithDefaultValue(name=f"Substring {i + 1}", type_=str, default_value="")
                  for i in range(number_elements)]
        output = TypedVariable(name="Concatenation", type_=str)
        Block.__init__(self, inputs=inputs, outputs=[output], name=name, position=position)

    def equivalent_hash(self):
        """ Custom hash function. Related to 'equivalent' method. """
        return self.number_elements + hash(self.separator)

    def equivalent(self, other):
        """ Returns whether the block is equivalent to the other given or not. """
        same_number = self.number_elements == other.number_elements
        same_separator = self.separator == other.separator
        return Block.equivalent(self, other) and same_number and same_separator

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
                 filename: str = "export", name: str = "", position: Tuple[float, float] = None):
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
        Block.__init__(self, inputs=inputs, outputs=[output], name=name, position=position)

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

        imports = [self.full_classname, full_classname(object_=self.method_type, compute_for='instance'),
                   full_classname(object_=self.method_type.class_, compute_for='class')]
        return ToScriptElement(declaration=script, imports=imports)


class Archive(Block):
    """
    A block that takes n inputs and store them in a archive (ZIP,...).

    :param number_exports: The number of files that will be stored in the archive
    :param filename: Name of the resulting archive file without its extension
    :param name: Name of the block.
    """

    def __init__(self, number_exports: int = 1, filename: str = "archive", name: str = "",
                 position: Tuple[float, float] = None):
        self.number_exports = number_exports
        self.filename = filename
        self.extension = "zip"
        self.text = False
        inputs = [Variable(name="export_" + str(i)) for i in range(number_exports)]
        inputs.append(TypedVariableWithDefaultValue(type_=str, default_value=filename, name="filename"))
        Block.__init__(self, inputs=inputs, outputs=[Variable(name="zip archive")], name=name, position=position)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Serialize the block with custom logic. """
        dict_ = Block.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)
        dict_['number_exports'] = len(self.inputs) - 1   # Filename is also a block input
        dict_["filename"] = self.filename
        return dict_

    @classmethod
    @set_block_variable_names_from_dict
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#'):
        """ Custom dict_to_object method. """
        return cls(number_exports=dict_["number_exports"], filename=dict_["filename"],
                   name=dict_['name'], position=dict_.get('position'))

    def evaluate(self, values, **kwargs):
        """ Generate archive stream for input streams. """
        name_input = self.inputs[-1]
        archive_name = f"{values.pop(name_input)}.{self.extension}"
        archive = BinaryFile(archive_name)
        with ZipFile(archive, 'w') as zip_archive:
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
