#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Gathers all workflow relative features. """

import ast
import time
import datetime
import uuid
from functools import cached_property
from typing import List, Union, Type, Any, Dict, Tuple, Optional, TypeVar, Callable
import warnings

import humanize
import psutil
import networkx as nx

import dessia_common.errors
from dessia_common.graph import get_column_by_node
from dessia_common.core import DessiaObject, display_settings_from_selector, stringify_dict_keys

from dessia_common.schemas.core import (get_schema, FAILED_ATTRIBUTE_PARSING, EMPTY_PARSED_ATTRIBUTE, SchemaStep,
                                        serialize_annotation, pretty_annotation, UNDEFINED, Schema, SchemaAttribute,
                                        SchemaAttributeGroup)
from dessia_common.utils.types import recursive_type, typematch, is_file_or_file_sequence
from dessia_common.utils.copy import deepcopy_value
from dessia_common.utils.diff import choose_hash
from dessia_common.utils.helpers import prettyname, is_sequence
from dessia_common.typings import JsonSerializable, ViewType
from dessia_common.files import StringFile, BinaryFile
from dessia_common.displays import DisplaySetting
from dessia_common.errors import SerializationError, ExtractionError
from dessia_common.warnings import SerializationWarning
from dessia_common.exports import ExportFormat, MarkdownWriter
import dessia_common.templates
from dessia_common.serialization import (deserialize, serialize_with_pointers, serialize, update_pointers_data,
                                         add_references, deserialize_argument)
from dessia_common.workflow.utils import (ToScriptElement, blocks_to_script, nonblock_variables_to_script,
                                          update_imports, generate_default_value)


T = TypeVar("T")

VariableAddress = Union[int, Tuple[int, int, int]]


class Variable(DessiaObject):
    """ New version of workflow variable. """

    _eq_is_data_eq = False

    def __init__(self, type_: Type[T] = None, default_value: T = UNDEFINED, name: str = "", label: str = "",
                 position: Tuple[float, float] = (0, 0)):
        self.default_value = default_value
        if type_ is None and self.has_default_value:
            type_ = type(default_value)
        self.type_ = type_
        self.label = label
        self.position = position
        self._locked = False
        super().__init__(name)

    @property
    def locked(self) -> bool:
        if self._locked is False:
            return False
        if self.has_default_value:
            return True
        raise ValueError(f"Variable '{self.name}' of type '{self.type_}' and labelled '{self.label}' "
                         f"is configured as 'Locked', but it does not implement a default value.")

    @locked.setter
    def locked(self, value: bool):
        raise AttributeError("Attribute 'locked' of Variable cannot be changed without ensuring a default value is "
                             "defined. Please use 'lock()' and 'unlock()' methods instead")

    def lock(self, value: T = UNDEFINED):
        """ Lock a variable while ensuring a default value is defined. """
        if value is UNDEFINED and not self.has_default_value:
            raise ValueError("Locking a Variable must implement a default value that is not UNDEFINED")

        if value is not UNDEFINED:
            self.default_value = value
        self._locked = True

    def unlock(self, keep_default: bool = True):
        """ Unlock a variable. If 'keep_default' is falsy, Variable's default value is reset to UNDEFINED. """
        if not keep_default:
            self.default_value = UNDEFINED
        self._locked = False

    @property
    def available_display_settings(self) -> List[DisplaySetting]:
        try:
            if issubclass(self.type_, DessiaObject):
                return self.type_.display_settings()
        except TypeError:
            pass
        return []

    def equivalent_hash(self):
        return len(self.name) + 7 * len(self.type_) + 11 * int(self.locked)

    def equivalent(self, other: 'Variable'):
        same_name = self.name == other.name
        same_type = self.type_ == other.type_
        same_lock = self.locked == other.locked
        return same_name and same_type and same_lock

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#',
                id_method=True, id_memo=None, **kwargs) -> JsonSerializable:
        """ Customize serialization method in order to handle undefined default value as well as pretty type. """
        dict_ = DessiaObject.base_dict(self)
        dict_.update({"type_": serialize_annotation(self.type_), "position": self.position,
                      "pretty_type": pretty_annotation(self.type_), "label": self.label, "locked": self.locked})
        if self.default_value is not UNDEFINED:
            dict_["default_value"] = serialize(self.default_value)
        return dict_

    def update_from_config(self, dict_):
        """ Set user config (label and default value) from workflow builder without serialization/deserialization. """
        default_value = dict_.get("default_value", UNDEFINED)
        self.default_value = deserialize(default_value)
        self.label = dict_.get("label", "")
        locked = dict_.get("locked")
        if locked:
            self.lock()

    @property
    def has_default_value(self):
        """ Helper property that indicates if default value should be trusted as such or is undefined. """
        return self.default_value is not UNDEFINED

    @cached_property
    def is_file_related(self) -> bool:
        """ Return whether a variable is of type File given its type_ attribute. """
        schema = get_schema(annotation=self.type_, attribute=SchemaAttribute(self.name))
        return schema.is_file_related

    def copy(self, deep: bool = False, memo=None):
        """
        Copy a Variable.

        :param deep: Deep copy if set to true, shallow copy if false. Defaults to False.
        :param memo: A memo that keeps track of already encountered objects, defaults to None.
        :return: A copy of the object
        """
        if memo is None:
            memo = {}
        default_value = deepcopy_value(self.default_value, memo=memo) if self.has_default_value else UNDEFINED
        variable = Variable(type_=self.type_, default_value=default_value, name=self.name)
        if self.locked:
            variable.lock()
        return variable

    def _to_script(self) -> ToScriptElement:
        script = self._get_to_script_elements()
        script.declaration = f"{self.__class__.__name__}({script.declaration})"

        script.imports.append(self.full_classname)
        return script

    def _get_to_script_elements(self):
        schema = get_schema(annotation=self.type_, attribute=SchemaAttribute(self.name))
        imports = schema.get_import_names(import_names=[])
        declaration = f"name='{self.name}', label='{self.label}', position={self.position}," \
                      f" type_={schema.pretty_annotation}"
        return ToScriptElement(declaration=declaration, imports=imports, imports_as_is=[])


RESULT_VARIABLE_NAME = "_result_name_"


NAME_VARIABLE = Variable(type_=str, name=RESULT_VARIABLE_NAME, label="Result Name")


class Block(DessiaObject):
    """ An Abstract block. Do not instantiate alone. """

    _eq_is_data_eq = False

    def __init__(self, inputs: List[Variable], outputs: List[Variable],
                 position: Tuple[float, float] = (0, 0), name: str = ''):
        self.inputs = inputs
        self.outputs = outputs
        self.position = position
        DessiaObject.__init__(self, name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#',
                id_method=True, id_memo=None, **kwargs) -> JsonSerializable:
        """ Overwrite default method to force use_pointers to true. """
        return super().to_dict(use_pointers=False, memo=memo, path=path, id_method=id_method, id_memo=id_memo, **kwargs)

    def equivalent_hash(self):
        """
        Custom hash of block that doesn't overwrite __hash__ as we do not want to lose python default equality behavior.

        Used by workflow module only.
        """
        return len(self.__class__.__name__) + 7 * len([i for i in self.inputs if i.locked])

    def equivalent(self, other: 'Block'):
        """
        Custom equal of block that does not overwrite __eq__ as we do not want to lose python default equality behavior.

        Used by workflow module only.
        """
        for input_, other_input in zip(self.inputs, other.inputs):
            if not input_.equivalent(other_input):
                return False
        return self.__class__.__name__ == other.__class__.__name__

    def _docstring(self):
        """ Base function for sub model docstring computing. """
        return {i: EMPTY_PARSED_ATTRIBUTE for i in self.inputs}

    def parse_input_doc(self, input_: Variable):
        """ Parse block docstring to get input documentation. """
        try:
            docstring = self._docstring()
            if input_ in docstring:
                return docstring[input_]
        except Exception:  # Broad except to avoid error 500 on doc computing
            return FAILED_ATTRIBUTE_PARSING
        return EMPTY_PARSED_ATTRIBUTE

    def base_script(self) -> str:
        """ Generate a chunk of script that denotes the arguments of a base block. """
        return f"name=\"{self.name}\", position={self.position}"

    def evaluate(self, values, **kwargs):
        """ Not implemented for abstract block class 'evaluate' method. """
        raise NotImplementedError("This method should be implemented in any Block inheriting class.")

    def _to_script(self, prefix: str):
        raise NotImplementedError("This method should be implemented in any Block inheriting class.")

    def is_valid(self, level: str = 'error') -> bool:  # TODO: Change this in further releases
        """ Always return True for now. """
        return True

    def deserialize_variables(self, dict_: JsonSerializable):
        """
        Enable inputs and outputs overwriting in order to allow input renaming as well as default value persistence.

        If no entry is given in dict, then we have default behavior with blocks generating their own inputs.
        """
        if "inputs" in dict_:
            for variable, config in zip(self.inputs, dict_["inputs"]):
                variable.update_from_config(config)
        if "outputs" in dict_:
            for variable, config in zip(self.outputs, dict_["outputs"]):
                variable.update_from_config(config)


class Pipe(DessiaObject):
    """
    Bind two variables of a Workflow.

    :param input_variable: The input variable of the pipe correspond to the start of the arrow, its tail.
    :param output_variable: The output variable of the pipe correspond to the end of the arrow, its hat.
    """

    _eq_is_data_eq = False

    def __init__(self, input_variable: Variable, output_variable: Variable, name: str = ''):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.memorize = False
        DessiaObject.__init__(self, name=name)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#', id_method=True, id_memo=None,
                **kwargs):
        """ Transform the pipe into a dict. """
        return {'input_variable': self.input_variable, 'output_variable': self.output_variable,
                'memorize': self.memorize}


class WorkflowError(Exception):
    """ Specific WorkflowError Exception. """


class Step:
    """ Step. """

    def __init__(self, label: str = "", inputs: List[Variable] = None, group_id: str = None, is_fallback: bool = False,
                 documentation: str = "", display_variable_index: int = None):
        self.label = label
        if inputs is None:
            inputs = []
        self.inputs = inputs
        if group_id is None:
            group_id = str(uuid.uuid4())
        self.group_id = group_id
        self.group_inputs = []
        self.display_setting = None
        self.is_fallback = is_fallback
        self.documentation = documentation
        self.display_variable_index = display_variable_index

    def __hash__(self):
        return (hash(self.label) + 43 * len(self.inputs) + 19 * len(self.group_inputs) + hash(self.display_setting)
                + len(self.documentation) + int(self.is_fallback) + self.display_variable_index)

    def __eq__(self, other: 'Step'):
        same_label = self.label == other.label
        same_inputs = all(i.equivalent(other_i) for i, other_i in zip(self.inputs, other.inputs))
        same_group = all(i.equivalent(other_i) for i, other_i in zip(self.group_inputs, other.group_inputs))
        same_display_setting = self.display_setting == other.display_setting
        same_documentation = len(self.documentation) == len(other.documentation)
        same_variable = self.display_variable_index == other.display_variable_index
        same_fallback = self.is_fallback is other.is_fallback
        return (same_label and same_inputs and same_group and same_display_setting and same_documentation
                and same_variable and same_fallback)

    def to_dict(self):
        """ Partial implementation of step dict. Inputs indices need to be added by parent workflow. """
        display_setting = self.display_setting.to_dict() if self.display_setting else None
        return {"label": self.label, "display_setting": display_setting, "inputs": [i.to_dict() for i in self.inputs],
                "group_inputs": [i.to_dict() for i in self.group_inputs], "group_id": self.group_id,
                "is_fallback": self.is_fallback, "documentation": self.documentation,
                "display_variable_index": self.display_variable_index}

    @classmethod
    def dict_to_object(cls, dict_, inputs: List[Variable], group_inputs: List[Variable]):
        step = cls(label=dict_["label"], inputs=inputs, group_id=dict_.get("group_id", None),
                   is_fallback=dict_.get("is_fallback", False), documentation=dict_.get("documentation", ""))
        step.group_inputs = group_inputs
        if dict_["display_setting"]:
            display_setting = DisplaySetting.dict_to_object(dict_["display_setting"])
            display_variable_index = dict_.get("display_variable_index", None)
            step.add_display_setting(display_setting=display_setting, inputs=group_inputs,
                                     variable_index=display_variable_index)
        return step

    def add_display_setting(self, display_setting: DisplaySetting, inputs: List[Variable], variable_index: int):
        self.display_setting = display_setting
        self.group_inputs = [i for i in inputs]
        self.display_variable_index = variable_index

    def remove_display_setting(self):
        self.display_setting = None
        self.group_inputs = []
        self.display_variable_index = None


class Workflow(Block):
    """
    Class Block of Workflows.

    :param blocks: A List with all the Blocks used by the Workflow.
    :param pipes: A List of Pipe objects.
    :param description: A short description that will be displayed on workflow card (frontend).
        Should be shorter than 100 chars
    :param documentation: A long documentation that will be displayed on workflow page (frontend).
        Can use markdown elements.
    :param name: The name of the workflow.
    """

    _standalone_in_db = True
    _eq_is_data_eq = True
    _allowed_methods = ["run", "start_run"]
    _non_serializable_attributes = ["branch_by_display_selector", "branch_by_export_format",
                                    "memorized_pipes", "coordinates", "detached_variables", "variables"]

    def __init__(self, blocks: List[Block], pipes: List[Pipe], output, *, steps: List[Step] = None,
                 detached_variables: List[Variable] = None, description: str = "", documentation: str = "",
                 name: str = ""):
        self.blocks = blocks
        self.pipes = pipes

        self.coordinates = {}

        self.variables = []
        self.nonblock_variables = []
        if detached_variables is None:
            detached_variables = []
        self.detached_variables = detached_variables
        for block in self.blocks:
            self.handle_block(block)

        for pipe in self.pipes:
            self.handle_pipe(pipe)

        self._cached_graph = None

        inputs = [v for v in self.variables if len(nx.ancestors(self.graph, v)) == 0]

        self.description = description
        self.documentation = documentation

        outputs = []
        self.output = output
        if output is not None:
            outputs.append(output)
            self._find_output_block(output)

        self._find_name()

        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

        self.branch_by_display_selector = self.branch_by_selector(self.display_blocks)
        self.branch_by_export_format = self.branch_by_selector(self.export_blocks)

        if steps:
            self.steps = steps
            step_inputs = [i for s in steps for i in s.inputs]
            self.spare_inputs = [i for i in self.inputs if i not in step_inputs]
        else:
            self.steps = []
            self.spare_inputs = [i for i in self.inputs]

    @classmethod
    def generate_empty(cls):
        """ Generate an empty workflow (mostly used by frontend to compute an init dict). """
        return cls(blocks=[], pipes=[], output=None)

    @property
    def nodes(self):
        """ Return the list of blocks and non_block_variables (nodes) of the Workflow. """
        return self.blocks + self.nonblock_variables

    @property
    def locked_inputs(self) -> List[Variable]:
        return [i for i in self.inputs if i.locked]

    @cached_property
    def file_inputs(self):
        """ Get all inputs that are files. """
        return [i for i in self.inputs if i.is_file_related]

    @cached_property
    def has_file_inputs(self) -> bool:
        """ Return True if there is any file input. """
        return any(self.file_inputs)

    @cached_property
    def memorized_pipes(self) -> List[Pipe]:
        """ Get pipes that are memorized. """
        return [p for p in self.pipes if p.memorize]

    def handle_pipe(self, pipe):
        """ Perform some initialization action on a pipe and its variables. """
        upstream_var = pipe.input_variable
        downstream_var = pipe.output_variable
        if upstream_var not in self.variables:
            self.variables.append(upstream_var)
            if upstream_var in self.detached_variables:
                self.detached_variables.remove(upstream_var)
            self.nonblock_variables.append(upstream_var)
        if downstream_var not in self.variables:
            self.variables.append(downstream_var)
            self.nonblock_variables.append(downstream_var)

    def handle_block(self, block):
        """ Perform some initialization action on a block and its variables. """
        if isinstance(block, Workflow):
            # Protecting direct Workflow blocks
            raise ValueError("Using workflow as blocks is forbidden, use WorkflowBlock wrapper instead")

        # Populating variables with block variables
        self.variables.extend(block.inputs)
        self.variables.extend(block.outputs)

        # Memorizing block incoming pipes
        if block in self.display_blocks:
            for i, input_ in enumerate(block.inputs):
                incoming_pipe = self.variable_input_pipe(input_)
                if incoming_pipe and i == block._displayable_input:
                    incoming_pipe.memorize = True

        try:
            self.coordinates[block] = (0, 0)
        except ValueError as err:
            raise ValueError(f"Cannot serialize block {block} ({block.name})") from err

    def _data_hash(self):
        output_hash = hash(self.variable_indices(self.output))
        base_hash = len(self.blocks) + 11 * len(self.pipes) + 23 * len(self.locked_inputs) + output_hash
        block_hash = int(sum(b.equivalent_hash() for b in self.blocks) % 1e6)
        step_hash = sum(hash(s) for s in self.steps)
        return (base_hash + block_hash + step_hash) % 1000000000

    def _data_eq(self, other_object) -> bool:
        if hash(self) != hash(other_object) or not Block.equivalent(self, other_object):
            return False

        # TODO: temp , reuse graph to handle block order!!!!
        for block1, block2 in zip(self.blocks, other_object.blocks):
            if not block1.equivalent(block2):
                return False

        if not self._equivalent_pipes(other_object):
            return False

        if not all(s == other_s for s, other_s in zip(self.steps, other_object.steps)):
            return False
        return True

    def _find_output_block(self, output: Variable):
        found_output = False
        i = 0
        while not found_output and i < len(self.blocks):
            found_output = output in self.blocks[i].outputs
            i += 1
        if not found_output:
            raise WorkflowError("workflow's output is not in any block's outputs")

    def _find_name(self):
        found_name = False
        i = 0
        all_nbvs = self.nonblock_variables + self.detached_variables
        while not found_name and i < len(all_nbvs):
            variable = all_nbvs[i]
            found_name = variable.name == RESULT_VARIABLE_NAME
            i += 1
        if not found_name:
            self.detached_variables.insert(0, NAME_VARIABLE)

    def _equivalent_pipes(self, other_wf) -> bool:
        pipes = []
        other_pipes = []
        for pipe, other_pipe in zip(self.pipes, other_wf.pipes):
            input_index = self.variable_index(pipe.input_variable)
            output_index = self.variable_index(pipe.output_variable)
            pipes.append((input_index, output_index))

            other_input_index = other_wf.variable_index(other_pipe.input_variable)
            other_output_index = other_wf.variable_index(other_pipe.output_variable)
            other_pipes.append((other_input_index, other_output_index))
        return set(pipes) == set(other_pipes)

    def __deepcopy__(self, memo=None):
        """ Return the deep copy. """
        if memo is None:
            memo = {}

        blocks = [deepcopy_value(b, memo) for b in self.blocks]
        output_adress = self.variable_indices(self.output)
        if output_adress is None:
            output = None
        else:
            output_block = blocks[output_adress[0]]
            output = output_block.outputs[output_adress[2]]

        # TODO Following code, while easily understandable, is not ideal.
        #  Instead of creating workflows with all blocks and pipes and stuff,
        #  we should generate them additevely, ie. crteate them as empty, then add blocks, then add pipes, and so on

        # Pipes need a workflow that has blocks in order to connect equivalent variables
        unpiped_workflow = Workflow(blocks=blocks, pipes=[], output=output, name=self.name)
        pipes = self.copy_pipes(unpiped_workflow)
        piped_workflow = Workflow(blocks=blocks, pipes=pipes, output=output, name=self.name)

        # Steps need a workflow that has pipe in order to handle NBVs and detached variables
        steps = [self.copy_step(copied_workflow=piped_workflow, step=s) for s in self.steps]
        workflow = Workflow(blocks=blocks, pipes=pipes, output=output, steps=steps, name=self.name)

        # Locking variables
        for i, input_ in enumerate(self.inputs):
            if input_.locked:
                workflow.inputs[i].lock(input_.default_value)
        return workflow

    def copy_step(self, copied_workflow: 'Workflow', step: Step) -> Step:
        inputs = []
        group_inputs = []
        for input_ in step.inputs:
            input_index = self.variable_indices(input_)
            copied_input = copied_workflow.variable_from_index(input_index)
            inputs.append(copied_input)
            if input_ in step.group_inputs:
                group_inputs.append(copied_input)
        copied_step = Step(label=step.label, inputs=inputs, is_fallback=step.is_fallback,
                           documentation=step.documentation, display_variable_index=step.display_variable_index)
        copied_step.group_inputs = group_inputs
        copied_step.display_setting = step.display_setting
        return copied_step

    def copy_pipe(self, copied_workflow: 'Workflow', pipe: Pipe) -> Pipe:
        """ Copy a single regular pipe. """
        upstream_index = self.variable_indices(pipe.input_variable)
        if self.is_variable_nbv(pipe.input_variable):
            raise dessia_common.errors.CopyError("copy_pipe method cannot handle nonblock-variables. "
                                                 "Please consider using copy_nbv_pipes")
        pipe_upstream = copied_workflow.variable_from_index(upstream_index)
        downstream_index = self.variable_indices(pipe.output_variable)
        pipe_downstream = copied_workflow.variable_from_index(downstream_index)
        return Pipe(pipe_upstream, pipe_downstream)

    def copy_nbv_pipe(self, copied_workflow: 'Workflow', pipe: Pipe, copy_memo: Dict[int, Variable]) -> Pipe:
        """
        Copy a pipe where its upstream variable is a NBV.

        This needs special care because if it is not handled properly, NBVs can duplicate,
        or copied pipes might be unordered.
        """
        nbv = pipe.input_variable
        upstream_index = self.variable_index(nbv)
        if upstream_index in copy_memo:
            copied_variable = copy_memo[upstream_index]
        else:
            copied_variable = nbv.copy()
            copy_memo[upstream_index] = copied_variable
        downstream_index = self.variable_indices(pipe.output_variable)
        pipe_downstream = copied_workflow.variable_from_index(downstream_index)
        return Pipe(copied_variable, pipe_downstream)

    def copy_pipes(self, copied_workflow: 'Workflow') -> List[Pipe]:
        """ Copy all pipes in workflow. """
        copy_memo = {}
        return [self.copy_nbv_pipe(copied_workflow=copied_workflow, pipe=p, copy_memo=copy_memo)
                if self.is_variable_nbv(p.input_variable)
                else self.copy_pipe(copied_workflow=copied_workflow, pipe=p)
                for p in self.pipes]

    def block_selector(self, block: Block) -> str:
        """ Get or create a generic selector for given block. """
        if block in self.display_blocks and block.selector:
            if isinstance(block.selector, ViewType):
                return block.selector.name
            if isinstance(block.selector, str):
                # Backward compatibility dessia_common < 0.14.0
                return block.selector
        return f"{self.selector_name(block)} ({self.blocks.index(block)})"

    def selector_name(self, block: Block) -> str:
        """ Compute name for selector. """
        name = block.name
        if name:
            return name
        if block in self.display_blocks:
            return block.type_
        if block in self.export_blocks:
            return block.extension
        return "Block"

    def branch_by_selector(self, blocks: List[Block]):
        """ Return the corresponding branch to each display or export selector. """
        selector_branches = {}
        for block in blocks:
            branch = self.secondary_branch_blocks(block)
            selector = self.block_selector(block)
            selector_branches[selector] = branch
        return selector_branches

    @property
    def display_blocks(self):
        """ Return list of blocks that can display something (3D, PlotData, Markdown,...). """
        return [b for b in self.blocks if hasattr(b, "_display_settings")]

    @property
    def blocks_display_settings(self) -> List[DisplaySetting]:
        """ Compute all display blocks display_settings. """
        display_settings = []
        for block in self.display_blocks:
            reference_path = "#"
            for i, input_ in enumerate(block.inputs):
                incoming_pipe = self.variable_input_pipe(input_)
                if i == block._displayable_input:
                    reference_path = f"{reference_path}/values/{self.pipes.index(incoming_pipe)}"
            block_index = self.blocks.index(block)
            settings = block._display_settings(block_index=block_index, reference_path=reference_path)
            if settings is not None:
                settings.selector = self.block_selector(block)
                display_settings.append(settings)
        return display_settings

    @classmethod
    def display_settings(cls, **kwargs) -> List[DisplaySetting]:
        """ Compute the displays settings of the workflow. """
        return [DisplaySetting(selector="Workflow", type_="workflow", method="to_dict"),
                DisplaySetting(selector="Documentation", type_="markdown", method="to_markdown"),
                DisplaySetting(selector="Tasks", type_="tasks", method="", load_by_default=True)]

    @property
    def export_blocks(self):
        """ Return list of blocks that can export something (3D, PlotData, Markdown,...). """
        return [b for b in self.blocks if hasattr(b, "_export_format")]

    @property
    def blocks_export_formats(self):
        """ Compute all export blocks export_formats. """
        export_formats = []
        for block in self.export_blocks:
            block_index = self.blocks.index(block)
            format_ = block._export_format(block_index)
            if format_ is not None:
                format_.selector = self.block_selector(block)
                export_formats.append(format_)
        return export_formats

    def _export_formats(self):
        """ Read block to compute available export formats. """
        export_formats = DessiaObject._export_formats(self)
        script_export = ExportFormat(selector="py", extension="py", method_name="save_script_to_stream", text=True)
        export_formats.append(script_export)
        return export_formats

    def to_markdown(self, **kwargs):
        """ Set workflow documentation as markdown. """
        return self.documentation

    def _docstring(self):
        """ Compute documentation of all blocks. """
        return [b._docstring() for b in self.blocks]

    def input_schema(self, input_: Variable, annotations: Dict[str, type]):
        input_index = self.input_index(input_)
        input_address = str(input_index)
        annotations[input_address] = input_.type_

        # Title & Description
        description = EMPTY_PARSED_ATTRIBUTE
        title = input_.label
        name = prettyname(input_.name)
        if input_ not in self.nonblock_variables and input_ not in self.detached_variables:
            block = self.block_from_variable(input_)
            description = block.parse_input_doc(input_)
            if not title:
                title = f"{block.name} - {name}"
        if input_ in self.nonblock_variables and not title:
            title = name
        return SchemaAttribute(name=input_address, default_value=input_.default_value, title=title,
                               editable=not input_.locked, documentation=description)

    @property
    def method_schemas(self):
        """ New support of method schemas. """
        steps = []
        for step in self.steps:
            attributes = []
            group_attributes = []
            annotations = {}
            for input_ in step.inputs:
                attribute = self.input_schema(input_=input_, annotations=annotations)
                if input_ in step.group_inputs:
                    group_attributes.append(attribute)
                else:
                    attributes.append(attribute)

            if group_attributes:
                group_name = step.display_setting.selector if step.display_setting else f"Group '{step.label}'"
                group_annotations = {a.name: annotations.pop(a.name) for a in group_attributes}
                attributes.append(SchemaAttributeGroup(annotations=group_annotations, attributes=group_attributes,
                                                       name=step.group_id, title=group_name))
            steps.append(SchemaStep(annotations=annotations, attributes=attributes, label=step.label,
                                    display_setting=step.display_setting, documentation=step.documentation,
                                    display_variable=step.display_variable_index))
        if self.spare_inputs:
            spare_annotations = {}
            spare_attributes = [self.input_schema(input_=i, annotations=spare_annotations) for i in self.spare_inputs]
            steps.append(SchemaStep(annotations=spare_annotations, attributes=spare_attributes, label="Default Step",
                                    is_fallback=True))

        schema = Schema(steps=steps, documentation=self.description)
        return {"run": schema.to_dict(method=True), "start_run": schema.to_dict(method=True, required=[])}

    def to_dict(self, use_pointers=False, memo=None, path="#", id_method=True, id_memo=None, **kwargs):
        """ Compute a dict from the object content. """
        dict_ = Block.to_dict(self, use_pointers=False)
        step_dicts = []
        for step in self.steps:
            step_dict = step.to_dict()
            inputs = [self.variable_indices(i) for i in step.inputs]
            group_inputs = [self.variable_indices(i) for i in step.group_inputs]
            step_dict.update({"inputs": inputs, "group_inputs": group_inputs})
            step_dicts.append(step_dict)
        dict_.update({"blocks": [b.to_dict(use_pointers=False) for b in self.blocks],
                      "pipes": [self.pipe_variable_indices(p) for p in self.pipes],
                      "output": self.variable_indices(self.output),
                      "nonblock_variables": [v.to_dict() for v in self.nonblock_variables + self.detached_variables],
                      "steps": step_dicts,
                      "package_mix": self.package_mix(),
                      "description": self.description,
                      "documentation": self.documentation})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs) -> 'Workflow':
        """ Recompute the object from a dict. """
        pointers_memo = kwargs.get("pointers_memo", None)
        global_dict = kwargs.get("global_dict", None)
        if pointers_memo is None or global_dict is None:
            global_dict, pointers_memo = update_pointers_data(global_dict=global_dict, current_dict=dict_,
                                                              pointers_memo=pointers_memo)

        init_workflow = initialize_workflow(dict_=dict_, global_dict=global_dict, pointers_memo=pointers_memo)

        if "imposed_variable_values" in dict_:
            # Backwards compatibility for locked inputs
            for variable_index_str, serialized_value in dict_["imposed_variable_values"].items():
                value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                variable = init_workflow.variable_from_index(ast.literal_eval(variable_index_str))
                variable.default_value = value
                variable.lock()

        # Following code is meh.
        if "steps" in dict_:
            steps = []
            # Backwards compatibility for steps
            for step_dict in dict_["steps"]:
                inputs = [init_workflow.variable_from_index(i) for i in step_dict["inputs"]]
                group_inputs = [init_workflow.variable_from_index(i) for i in step_dict["group_inputs"]]
                step = Step.dict_to_object(step_dict, inputs, group_inputs)
                steps.append(step)
        else:
            steps = None
        return cls(blocks=init_workflow.blocks, pipes=init_workflow.pipes, output=init_workflow.output,
                   steps=steps, description=dict_.get("description", ""),
                   documentation=dict_.get("documentation", ""), name=dict_["name"])

    def dict_to_arguments(self, dict_: JsonSerializable, method: str, global_dict=None, pointers_memo=None, path='#'):
        """ Process a JSON of arguments and deserialize them. """
        dict_ = {int(k): v for k, v in dict_.items()}  # Serialization set keys as strings
        if method in self._allowed_methods:
            name = None
            arguments_values = {}
            for i, input_ in enumerate(self.inputs):
                overwritten = input_.has_default_value and i in dict_
                if (not input_.has_default_value or overwritten) and not input_.locked:
                    path_value = f"{path}/inputs/{i}"
                    value = deserialize_argument(type_=input_.type_, argument=dict_[i], global_dict=global_dict,
                                                 pointers_memo=pointers_memo, path=path_value)
                    if input_.name == RESULT_VARIABLE_NAME:
                        name = value
                    arguments_values[i] = value
            if name is None and len(self.inputs) in dict_ and isinstance(dict_[len(self.inputs)], str):
                # Hot fixing name not attached
                name = dict_[len(self.inputs)]
            return {"input_values": arguments_values, "name": name}
        raise NotImplementedError(f"Method '{method}' is not allowed for Workflow. Expected 'run' or 'start_run'.")

    @property
    def default_values(self):
        return {i: input_.default_value for i, input_ in enumerate(self.inputs) if input_.has_default_value}

    def _run_dict(self) -> JsonSerializable:
        return {str(self.variables.index(i)): serialize(v) for i, v in self.default_values.items()}

    def _start_run_dict(self) -> Dict:
        return {}

    def method_dict(self, method_name: str = None) -> Dict:
        """ Wrapper method to get dictionaries of run and start_run methods. """
        if method_name == 'run':
            return self._run_dict()
        if method_name == 'start_run':
            return self._start_run_dict()
        raise WorkflowError(f"Calling method_dict with unknown method_name '{method_name}'")

    def variable_from_index(self, index: VariableAddress):
        """ Index elements are, in order : (Block index : int, Port side (0: input, 1: output), Port index : int). """
        if isinstance(index, int):
            return self.nonblock_variables[index]
        if not index[1]:
            return self.blocks[index[0]].inputs[index[2]]
        return self.blocks[index[0]].outputs[index[2]]

    def _get_graph(self):
        """ Cached property for graph. """
        if self._cached_graph is None:
            self._cached_graph = self._graph()
        return self._cached_graph

    graph = property(_get_graph)

    def _graph(self):
        """ Compute the networkx graph of the workflow. """
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
        """ Return blocks that are upstream for output. """
        # TODO Check what's happening when output is null (incomplete workflow)
        output_block = self.block_from_variable(self.output)
        output_upstreams = self.upstream_blocks(output_block)
        runtime_blocks = [output_block] + output_upstreams
        i = 0
        while output_upstreams and i <= len(self.blocks):
            block_upstreams = []
            for block in output_upstreams:
                block_upstreams.extend(self.upstream_blocks(block))
            output_upstreams = block_upstreams
            for candidate in block_upstreams:
                if candidate not in runtime_blocks:
                    runtime_blocks.append(candidate)
            i += 1
        return runtime_blocks

    def is_block_secondary(self, block: Block) -> bool:
        """ Return whether given block is part of the main branch or not. """
        return block not in self.runtime_blocks

    def has_upstream_block(self, block: Block) -> bool:
        """ Return whether given block has any upstream block or not. """
        return bool(self.upstream_blocks(block))

    def _branch_blocks(self, block: Block, condition: Callable[[Block], bool], include_last: bool = False):
        upstream_blocks = self.upstream_blocks(block)
        branch_blocks = [block]
        i = 0
        candidates = upstream_blocks
        while candidates and i <= len(self.blocks):
            candidates = []
            for upstream_block in upstream_blocks:
                if condition(upstream_block):
                    branch_blocks.insert(0, upstream_block)
                    candidates.extend(self.upstream_blocks(upstream_block))
                elif include_last:
                    branch_blocks.insert(0, upstream_block)
            upstream_blocks = [b for b in candidates if b not in branch_blocks]
            i += 1
        return branch_blocks

    def secondary_branch_blocks(self, block: Block) -> List[Block]:
        """
        Compute the necessary upstream blocks to run a part of a workflow that leads to the given block.

        It stops looking for blocks when it reaches the main branch, and memorize the connected pipe.

        :param block: Block that is the target of the secondary branch
        """
        branch_blocks = self._branch_blocks(block, self.is_block_secondary)
        for branch_block in branch_blocks:
            upstream_blocks = self.upstream_blocks(branch_block)
            for upstream_block in upstream_blocks:
                if upstream_block in self.runtime_blocks:
                    for pipe in self.pipes_between_blocks(upstream_block, branch_block):
                        pipe.memorize = True
        return branch_blocks

    def pipe_from_variable_indices(self, upstream_indices: VariableAddress,
                                   downstream_indices: VariableAddress) -> Pipe:
        """ Get a pipe from the global indices of its attached variables. """
        for pipe in self.pipes:
            if self.variable_indices(pipe.input_variable) == upstream_indices \
                    and self.variable_indices(pipe.output_variable) == downstream_indices:
                return pipe
        msg = f"No pipe has '{upstream_indices}' as upstream variable and '{downstream_indices}' as downstream variable"
        raise ValueError(msg)

    def pipe_variable_indices(self, pipe: Pipe) -> Tuple[VariableAddress, VariableAddress]:
        """ Return the global indices of a pipe's attached variables. """
        return self.variable_indices(pipe.input_variable), self.variable_indices(pipe.output_variable)

    def variable_input_pipe(self, variable: Variable) -> Optional[Pipe]:
        """ Get the incoming pipe for a variable. If variable is not connected, returns None. """
        incoming_pipes = [p for p in self.pipes if p.output_variable == variable]
        if incoming_pipes:  # Inputs can only be connected to one pipe
            incoming_pipe = incoming_pipes[0]
            return incoming_pipe
        return None

    def variable_output_pipes(self, variable: Variable) -> List[Optional[Pipe]]:
        """ Compute all pipes going out a given variable. """
        return [p for p in self.pipes if p.input_variable == variable]

    def pipes_between_blocks(self, upstream_block: Block, downstream_block: Block):
        """ Compute all the pipes linking two blocks. """
        pipes = []
        for outgoing_pipe in self.block_outgoing_pipes(upstream_block):
            if outgoing_pipe is not None and outgoing_pipe in self.block_incoming_pipes(downstream_block):
                pipes.append(outgoing_pipe)
        return pipes

    def block_incoming_pipes(self, block: Block) -> List[Optional[Pipe]]:
        """ Get incoming pipes for every block variable. """
        return [self.variable_input_pipe(i) for i in block.inputs]

    def block_outgoing_pipes(self, block: Block) -> List[Pipe]:
        """ Return all block outgoing pipes. """
        outgoing_pipes = []
        for output in block.outputs:
            outgoing_pipes.extend(self.variable_output_pipes(output))
        return outgoing_pipes

    def block_upstream_variables(self, block: Block):
        """
        Return block inputs configuration.

        Inputs can be of different states :
        - available : input has no upstream (it is not connected by a pipe) and is not locked -> references itself
        - nonblock : input has an upstream, which is a nonblock variable, and is not locked -> references the NBV
        - wired : input is connected by a pipe to another block output -> references upstream block output
        - locked : input or upstream NBV is locked -> references itself or the NBV
        """
        upstream_variables = {"available": [], "nonblock": [], "wired": [], "locked": []}
        input_upstreams = [self.upstream_variable(i) for i in block.inputs]
        for upstream_variable, block_input in zip(input_upstreams, block.inputs):
            if upstream_variable is None:
                category = "locked" if block_input.locked else "available"
                upstream_variables[category].append(block_input)
            elif upstream_variable in self.nonblock_variables:
                category = "locked" if upstream_variable.locked else "nonblock"
                upstream_variables[category].append(upstream_variable)
            else:
                upstream_variables["wired"].append(upstream_variable)
        return upstream_variables

    def upstream_blocks(self, block: Block) -> List[Block]:
        """ Return a list of given block's upstream blocks. """
        upstream_variables = self.block_upstream_variables(block)
        upstream_blocks = [self.block_from_variable(v) for v in upstream_variables["wired"]]
        return list(set(upstream_blocks))

    def upstream_branch(self, block: Block) -> List[Block]:
        return self._branch_blocks(block=block, condition=self.has_upstream_block, include_last=True)

    def get_upstream_nbv(self, variable: Variable) -> Variable:
        """ If given variable has an upstream nonblock_variable, return it otherwise return given variable itself. """
        if not self.nonblock_variables:
            return variable
        upstream_variable = self.upstream_variable(variable)
        if upstream_variable is not None and upstream_variable in self.nonblock_variables:
            return upstream_variable
        return variable

    def upstream_variable(self, variable: Variable) -> Optional[Variable]:
        """
        Return upstream variable if given variable is connected to a pipe as a pipe output.

        :param variable: Variable to search an upstream for
        """
        incoming_pipe = self.variable_input_pipe(variable)
        if incoming_pipe:
            return incoming_pipe.input_variable
        return None

    def upstream_inputs(self, variable: Variable) -> List[Variable]:
        """
        Return all upstream inputs needed to reach given variables.

        :param variable: Variable to reach
        """
        block = self.block_from_variable(variable)
        upstream_inputs = []
        branch_blocks = self.upstream_branch(block)
        for block in branch_blocks:
            upstream_variables = self.block_upstream_variables(block)
            upstream_inputs.extend(upstream_variables["available"] + upstream_variables["nonblock"])
        return list(set(upstream_inputs))

    def variable_indices(self, variable: Variable) -> Optional[Union[Tuple[int, int, int], int]]:
        """
        Return global address of given variable as a tuple or an int.

        If variable is non block, return index of variable in variables sequence
        Else returns global address (index_block, index, index_port)
        """
        if variable is None:
            return None

        for iblock, block in enumerate(self.blocks):
            if variable in block.inputs:
                return iblock, 0, block.inputs.index(variable)
            if variable in block.outputs:
                return iblock, 1, block.outputs.index(variable)

        upstream_variable = self.get_upstream_nbv(variable)
        if upstream_variable in self.nonblock_variables:
            # Free variable not attached to block
            return self.nonblock_variables.index(upstream_variable)
        raise WorkflowError(f"Something is wrong with variable {variable.name}")

    def is_variable_nbv(self, variable: Variable) -> bool:
        """ Return True if variable does not belong to a block. """
        return isinstance(self.variable_indices(variable), int)

    def block_from_variable(self, variable) -> Block:
        """ Return block of which given variable is attached to. """
        iblock, _, _ = self.variable_indices(variable)
        return self.blocks[iblock]

    def output_disconnected_elements(self):
        """ Return blocks and variables that are not attached to the output. """
        disconnected_elements = []
        ancestors = nx.ancestors(self.graph, self.output)
        for block in self.blocks:
            if block not in ancestors:
                disconnected_elements.append(block)

        for variable in self.nonblock_variables:
            if variable not in ancestors:
                disconnected_elements.append(variable)
        return disconnected_elements

    def input_index(self, variable: Variable) -> Optional[int]:
        """ If variable is a workflow input, returns its index. """
        upstream_variable = self.get_upstream_nbv(variable)
        if upstream_variable in self.inputs:
            return self.inputs.index(upstream_variable)
        return None

    def variable_index(self, variable: Variable) -> int:
        """ Return variable index in variables sequence. """
        return self.variables.index(variable)

    def block_inputs_global_indices(self, block_index: int) -> List[int]:
        """ Return given block inputs global indices in inputs sequence. """
        block = self.blocks[block_index]
        indices = [self.input_index(i) for i in block.inputs]
        return [i for i in indices if i is not None]

    def match_variables(self, serialize_output: bool = False):
        """
        Run a check for every variable to find its matchable counterparts.

        This means :
            - Variables are compatible workflow-wise
            - Their types are compatible
        """
        variable_match = {}
        for variable in self.variables:
            if variable.type_ is not None:
                vartype = variable.type_
            else:
                continue
            if serialize_output:
                varkey = str(self.variable_indices(variable))
            else:
                varkey = variable
            variable_match[varkey] = []
            for other_variable in self.variables:
                if not self.variable_compatibility(variable, other_variable):
                    continue
                other_vartype = other_variable.type_
                if typematch(vartype, other_vartype):
                    if serialize_output:
                        varval = str(self.variable_indices(other_variable))
                    else:
                        varval = other_variable
                    variable_match[varkey].append(varval)
        return variable_match

    def variable_compatibility(self, variable: Variable, other_variable: Variable) -> bool:
        """
        Check compatibility between variables.

        Two variables are compatible if :
            - They are not equal
            - They don't share the same block
            - They are not input/input or output/output
            - They are typed
        """
        if variable == other_variable:  # If this is the same variable, it is not compatible
            return False

        adress = self.variable_indices(variable)
        other_adress = self.variable_indices(other_variable)

        if variable not in self.nonblock_variables and other_variable not in self.nonblock_variables:
            # If both aren't NBVs we need to check more non-equality elements
            same_block = adress[0] == other_adress[0]
            same_side = adress[1] == other_adress[1]
            if same_block or same_side:
                # A variable cannot be compatible with one on a same block
                # or being the same side (input/input, output/output)
                return False
        # If both are NBVs, non-equality has already been checked
        # If one is NBV and not the other, there is no need to check non-equality

        if variable.type_ is None or other_variable.type_ is None:
            # Variable must be typed to be seen compatible
            return False
        return True

    @property
    def layout_graph(self) -> nx.DiGraph:
        """ Compute graph layout. """
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)  # does not handle detached_variable

        for pipe in self.pipes:
            if pipe.input_variable in self.nonblock_variables:
                input_node = pipe.input_variable
            else:
                input_node = self.block_from_variable(pipe.input_variable)
            output_block = self.block_from_variable(pipe.output_variable)
            graph.add_edge(input_node, output_block)
        return graph

    def graph_columns(self, graph):
        """
        Store nodes of a workflow into a list of nodes indexes.

        :returns: A list of Column Layout where a Column Layout is a list of node_indices
        """
        column_by_node = get_column_by_node(graph)
        nodes_by_column = {}
        for node, column_index in column_by_node.items():
            node_index = self.nodes.index(node)
            nodes_by_column[column_index] = nodes_by_column.get(column_index, []) + [node_index]
        return list(nodes_by_column.values())

    def layout(self):
        """
        Stores a workflow graph layout.

        :returns: A list of Graph Layouts where a GraphLayout is a list of Column Layout
                and Column Layout a list node_indices.
        """
        digraph = self.layout_graph
        graph = digraph.to_undirected()
        connected_components = nx.connected_components(graph)
        return [self.graph_columns(digraph.subgraph(cc)) for cc in list(connected_components)]

    def plot_graph(self):
        """ Plot graph by means of networking and matplotlib. """
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

    def input_values(self, user_values=None):
        if user_values is None:
            user_values = {}
        allowed_values = {i: v for i, v in user_values.items() if not self.inputs[i].locked}
        input_values = self.default_values
        input_values.update(allowed_values)
        return input_values

    def run(self, input_values=None, verbose=False, progress_callback=lambda x: None, name=None):
        """ Full run of a workflow. Yields a WorkflowRun. """
        log = ""

        values = self.input_values(input_values)
        state = self.start_run(values)
        state.activate_inputs(check_all_inputs=True)

        start_time = time.time()
        start_timestamp = datetime.datetime.now()

        log_msg = "Starting workflow run at {}"
        log_line = log_msg.format(time.strftime("%d/%m/%Y %H:%M:%S UTC", time.gmtime(start_time)))
        log += (log_line + "\n")
        if verbose:
            print(log_line)

        state.continue_run(progress_callback=progress_callback)

        end_time = time.time()
        log_line = f"Workflow terminated in {end_time - start_time} s"

        log += log_line + '\n'
        if verbose:
            print(log_line)

        if not name:
            timestamp = start_timestamp.strftime("%m-%d (%H:%M)")
            name = f"{self.name} @ [{timestamp}]"
        return state.to_workflow_run(name=name)

    def start_run(self, input_values=None, name: str = None):
        """ Partial run of a workflow. Yields a WorkflowState. """
        return WorkflowState(self, input_values=input_values, name=name)

    def is_valid(self, level: str = "error"):
        """ Tell if the workflow is valid by checking type compatibility of pipes inputs/outputs. """
        for pipe in self.pipes:
            upstream = pipe.input_variable.type_
            downstream = pipe.output_variable.type_
            if upstream.type_ != downstream.type_:
                try:
                    issubclass(upstream.type_.type_, downstream.type_)
                except TypeError as error:  # TODO: need of a real typing check
                    consistent = True
                    if not consistent:
                        raise TypeError(f"Inconsistent pipe type from pipe input '{upstream.name}'"
                                        f"to pipe output '{downstream.name}': "
                                        f"'{upstream.type_}' incompatible with '{downstream.type_}'") from error
        return True

    def package_mix(self) -> Dict[str, float]:
        """ Compute a structure showing percentages of packages used. """
        package_mix = {}
        for block in self.blocks:
            if hasattr(block, 'package_mix'):
                for package_name, fraction in block.package_mix().items():
                    if package_name in package_mix:
                        package_mix[package_name] += fraction
                    else:
                        package_mix[package_name] = fraction

        # Make dimensionless
        fraction_sum = sum(package_mix.values())
        return {pn: f / fraction_sum for pn, f in package_mix.items()}

    def _to_script(self, prefix: str = '') -> ToScriptElement:
        """
        Computes elements for a to_script interpretation.

        :returns: ToSriptElement
        """
        workflow_output_index = self.variable_indices(self.output)
        if workflow_output_index is None:
            raise ValueError("A workflow output must be set")

        imports = []
        imports_as_is = []
        # --- Blocks ---
        blocks_str = blocks_to_script(blocks=self.blocks, prefix=prefix, imports=imports)

        # --- NBVs ---
        nbvs_str = nonblock_variables_to_script(nonblock_variables=self.nonblock_variables, prefix=prefix,
                                                imports=imports, imports_as_is=imports_as_is)

        # --- Pipes ---
        pipes_str = self.pipes_to_script(prefix=prefix, imports=imports)

        # --- Building script ---
        output_name = f"{prefix}block_{workflow_output_index[0]}.outputs[{workflow_output_index[2]}]"

        full_script = f"{blocks_str}\n" \
                      f"{nbvs_str}\n" \
                      f"{pipes_str}\n" \
                      f"{prefix}workflow = " \
                      f"Workflow({prefix}blocks, {prefix}pipes, output={output_name}, documentation=documentation," \
                      f" name=\"{self.name}\")\n"

        imports.append(self.full_classname)
        full_script = f'documentation = """{self.documentation}"""\n\n' + full_script
        return ToScriptElement(declaration=full_script, imports=imports, imports_as_is=imports_as_is)

    def to_script(self) -> str:
        """ Compute a script representing the workflow. """
        workflow_output_index = self.variable_indices(self.output)
        if workflow_output_index is None:
            raise ValueError("A workflow output must be set")

        self_script = self._to_script()

        script_imports = self_script.imports_to_str()

        return f"{script_imports}\n" \
               f"{self_script.declaration}"

    def pipes_to_script(self, prefix, imports):
        """ Set pipes to script. """
        if len(self.pipes) > 0:
            imports.append(self.pipes[0].full_classname)

        pipes_str = ""
        pipe_names = []
        for ipipe, pipe in enumerate(self.pipes):
            pipe_name = f"{prefix}pipe_{ipipe}"
            input_index = self.variable_indices(pipe.input_variable)
            if self.is_variable_nbv(pipe.input_variable):  # NBV handling
                input_name = f'{prefix}variable_{input_index}'
            else:
                input_name = f"{prefix}block_{input_index[0]}.outputs[{input_index[2]}]"

            output_index = self.variable_indices(pipe.output_variable)
            if self.is_variable_nbv(pipe.output_variable):  # NBV handling
                output_name = f'{prefix}variable_{output_index}'
            else:
                output_name = f"{prefix}block_{output_index[0]}.inputs[{output_index[2]}]"
            pipes_str += f"{pipe_name} = Pipe({input_name}, {output_name})\n"
            pipe_names.append(pipe_name)
        pipes_str += f"{prefix}pipes = [{', '.join(pipe_names)}]\n"
        return pipes_str

    def save_script_to_stream(self, stream: StringFile):
        """ Save the workflow to a python script to a stream. """
        string = self.to_script()
        stream.seek(0)
        stream.write(string)

    def save_script_to_file(self, filename: str):
        """ Save the workflow to a python script to a file on the disk. """
        if not filename.endswith('.py'):
            filename += '.py'
        with open(filename, 'w', encoding='utf-8') as file:
            self.save_script_to_stream(file)

    def evaluate(self, values, **kwargs):
        """ Not implemented Workflow as Block evaluate method. """
        raise NotImplementedError("Method 'evaluate' is not implemented for class Workflow.")

    @property
    def displayable_outputs(self):
        return [o for b in self.blocks for o in b.outputs if o.available_display_settings]

    def displayable_display_settings(self):
        """Callable from frontend."""
        missing_inputs = []
        available_display_settings = []
        ds_inputs = {
            output: [self.inputs.index(upstream_input) for upstream_input in self.upstream_inputs(output)]
            for output in self.displayable_outputs
        }
        outputs_labels = {}
        for step in reversed(self.steps):
            if not step.is_fallback:
                step_display_settings = {}
                for output, indexes in ds_inputs.items():
                    if all(index not in missing_inputs for index in indexes):
                        output_index = self.variables.index(output)
                        step_display_settings[output_index] = [
                            ds.to_dict() for ds in output.available_display_settings
                        ]
                        outputs_labels[output_index] = output.label
                available_display_settings.append(step_display_settings)
                missing_inputs.extend(self.inputs.index(step_input) for step_input in step.inputs)
        return {"available_display_settings": available_display_settings[::-1],"outputs_labels": outputs_labels}

    def change_input_step(self, input_index: int, new_step_index: int):
        input_ = self.inputs[input_index]
        current_step = self.find_input_step(input_)
        new_step = self.steps[new_step_index]
        new_step.inputs.append(input_)
        current_collection = self.spare_inputs if current_step is None else current_step.inputs
        current_collection.remove(input_)

    def change_inputs_step(self, input_indices: List[int], new_step_index: int):
        """ Callable from frontend. """
        for input_index in input_indices:
            self.change_input_step(input_index, new_step_index)
        return self.method_schemas["run"]

    def remove_input_from_step(self, input_index: int):
        """ Callable from frontend. """
        input_ = self.inputs[input_index]
        self.spare_inputs.append(input_)
        current_step = self.find_input_step(input_)
        current_step.inputs.remove(input_)
        return self.method_schemas["run"]

    def reorder_step_inputs(self, step_index: int, order: list[int]):
        """ Callable from frontend. """
        step = self.steps[step_index]
        step_input_indices = [self.input_index(v) for v in step.inputs]
        if set(step_input_indices) != set(order):
            raise ValueError(f"Reordering inputs of step '{step.label}' failed. "
                             f"Given order '{order}' does not match step inputs of indices '{step_input_indices}'")
        step.inputs = [self.inputs[i] for i in order]
        return self.method_schemas["run"]

    def insert_step(self, label: str = "", step_index: int = None):
        """ Callable from frontend. """
        step = Step(label=label)
        if step_index is None:
            self.steps.append(step)
        else:
            self.steps.insert(step_index, step)
        return self.method_schemas["run"]

    def rename_step(self, step_index: int, label: str = ""):
        """ Callable from frontend. """
        step = self.steps[step_index]
        step.label = label
        return self.method_schemas["run"]

    def reorder_steps(self, order: list[int]):
        """ Callable from frontend. """
        if len(order) != len(self.steps):
            raise ValueError(f"Given order '{order}' of length '{len(order)}' is missing items."
                             f"It should be of length '{len(self.steps)}' ")
        if len(set(order)) != len(self.steps):
            raise ValueError(f"Duplicated indices found in given order '{order}'")
        self.steps = [self.steps[i] for i in order]
        return self.method_schemas["run"]

    def remove_step(self, step_index: int):
        """ Callable from frontend. """
        step = self.steps[step_index]
        for input_ in reversed(step.inputs):
            input_index = self.inputs.index(input_)
            self.remove_input_from_step(input_index)
        del self.steps[step_index]
        return self.method_schemas["run"]

    def reset_steps(self):
        """ Callable from frontend. """
        for i in reversed(range(len(self.steps))):
            self.remove_step(i)
        return self.method_schemas["run"]

    def find_input_step(self, input_: Variable) -> Optional[Step]:
        steps = [s for s in self.steps if input_ in s.inputs]
        if not len(steps):
            return None
        if len(steps) == 1:
            return steps[0]
        raise ValueError(f"Input '{input_.name}', labelled '{input_.label}' found twice in steps.")

    def add_step_display(self, variable_index: int, step_index: int, selector: str):
        """ Callable from frontend. """
        variable = self.variables[variable_index]
        upstream_inputs = self.upstream_inputs(variable)
        previous_step_inputs = [input_ for step in self.steps[:step_index] for input_ in step.inputs]
        for input_ in upstream_inputs:
            if input_ in previous_step_inputs:
                continue
            index = self.input_index(input_)
            self.change_input_step(input_index=index, new_step_index=step_index)
        step = self.steps[step_index]
        display_setting = display_settings_from_selector(display_settings=variable.available_display_settings,
                                                         selector=selector)
        step.add_display_setting(display_setting=display_setting, inputs=upstream_inputs, variable_index=variable_index)
        return self.method_schemas["run"]

    def remove_step_display(self, step_index: int):
        """ Callable from frontend. """
        step = self.steps[step_index]
        step.remove_display_setting()
        return self.method_schemas["run"]

    def compute_step_display(self, step_index: int, dict_):
        """
        Callable from frontend.

        Compute the branch that is needed to run in order to generate the output whose display is being display
        on frontend.

        :param step_index: The index of the step that needs to update its current view based on dict_ argument.
        :param dict_: The dict_ of needed inputs given by users from frontend.
        :return: The display dict
        """
        step = self.steps[step_index]
        display_setting = step.display_setting
        if display_setting is None:
            return None
        variable = self.variables[step.display_variable_index]
        block_index, variable_type, output_index = self.variable_indices(variable)
        if variable_type == 0:
            raise ValueError("Given variable is an input and cannot be used to display in a form")
        block = self.blocks[block_index]
        branch = self.upstream_branch(block)
        deserialized_dict = {int(i): deserialize(v) for i, v in dict_.items()}
        input_values = self.input_values(deserialized_dict)
        workflow_state = self.start_run(input_values=input_values)
        block_args = {b: {} for b in branch}
        evaluated_blocks = workflow_state.evaluate_branch(blocks=branch, block_args=block_args)
        output = evaluated_blocks[block][output_index]
        display_object = output._display_from_selector(display_setting.selector)
        return stringify_dict_keys(display_object.to_dict())

    def save_step_documentation(self, step_index: int, documentation: str):
        """ Callable from frontend. """
        step = self.steps[step_index]
        step.documentation = documentation
        return self.method_schemas["run"]

    def rename_input(self, input_index: int, label: str):
        """ Callable from frontend. """
        input_ = self.inputs[input_index]
        input_.label = label
        return self.method_schemas["run"]

    def lock_input(self, input_index: int, default_value=UNDEFINED):
        """ Callable from frontend. """
        input_ = self.inputs[input_index]
        input_.lock(default_value)
        return self.method_schemas["run"]

    def set_input_default(self, input_index: int, value):
        """ Callable from frontend. """
        input_ = self.inputs[input_index]
        input_.default_value = value
        return self.method_schemas["run"]

    def reset_input_default(self, input_index: int):
        return self.set_input_default(input_index=input_index, value=UNDEFINED)


class ExecutionInfo(DessiaObject):
    """ Workflow execution information: start & end date, memory consumption. """

    def __init__(self, start_time: float = None, end_time: float = None,
                 before_block_memory_usage: List[Tuple[Block, int]] = None,
                 after_block_memory_usage: List[Tuple[Block, int]] = None):

        if start_time is None:
            start_time = time.time()

        self.start_time = start_time
        self.end_time = end_time

        if before_block_memory_usage is None:
            before_block_memory_usage = []
        self.before_block_memory_usage = before_block_memory_usage

        if after_block_memory_usage is None:
            after_block_memory_usage = []
        self.after_block_memory_usage = after_block_memory_usage

        DessiaObject.__init__(self, name="")

    @property
    def execution_time(self):
        """ Computes execution time. May return None if end_time is. """
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#',
                id_method=True, id_memo=None, **kwargs):
        """ Serialize the ExecutionInfo. """
        dict_ = {"start_time": self.start_time, "end_time": self.end_time}
        block_indices = kwargs['block_indices']
        dict_["before_block_memory_usage"] = [(block_indices[b], m) for b, m in self.before_block_memory_usage]
        dict_["after_block_memory_usage"] = [(block_indices[b], m) for b, m in self.after_block_memory_usage]
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, **kwargs):
        """ Deserialize the ExecutionInfo. """
        index_to_block = kwargs['index_to_block']
        before_block_memory_usage = [(index_to_block[int(i)], m) for i, m in dict_["before_block_memory_usage"]]
        after_block_memory_usage = [(index_to_block[int(i)], m) for i, m in dict_["after_block_memory_usage"]]
        return cls(start_time=dict_["start_time"], end_time=dict_["end_time"],
                   before_block_memory_usage=before_block_memory_usage,
                   after_block_memory_usage=after_block_memory_usage)

    def to_markdown(self, **kwargs):
        """ Render to markdown. Require blocks for clean order. """
        table_content = []
        for block, mem_start in dict(self.before_block_memory_usage).items():
            mem_end = dict(self.after_block_memory_usage)[block]
            mem_diff = mem_end - mem_start
            table_content.append((block.name, f"{humanize.naturalsize(mem_start)}", f"{humanize.naturalsize(mem_end)}",
                                  f"{humanize.naturalsize(mem_diff)}"))
        writer = MarkdownWriter(print_limit=25, table_limit=None)
        return writer.matrix_table(table_content,
                                   ["Block", "Memory usage at start", "Memory usage at end", "Memory diff"])


class WorkflowState(DessiaObject):
    """ State of execution of a workflow. """

    _standalone_in_db = True
    _allowed_methods = ['block_evaluation', 'evaluate_next_block', 'continue_run',
                        'evaluate_maximum_blocks', 'add_block_input_values']
    _non_serializable_attributes = ['activated_items']

    def __init__(self, workflow: Workflow, input_values=None, activated_items=None, values=None,
                 output_value=None, log: str = '', execution_info: ExecutionInfo = None, name: str = ''):
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

        if execution_info is None:
            execution_info = ExecutionInfo()
        self.execution_info = execution_info

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
        for pipe, value in self.values.items():
            variable_indices = self.workflow.pipe_variable_indices(pipe)
            copied_pipe = workflow.pipe_from_variable_indices(*variable_indices)
            values[copied_pipe] = value

        activated_items = {}
        for item, value in self.activated_items.items():
            if isinstance(item, Variable):
                copied_item = workflow.variable_from_index(self.workflow.variable_indices(item))
            elif isinstance(item, Block):
                copied_item = workflow.blocks[self.workflow.blocks.index(item)]
            elif isinstance(item, Pipe):
                copied_item = workflow.pipes[self.workflow.pipes.index(item)]
            else:
                raise ValueError(f"WorkflowState Copy Error : item {item} cannot be activated")
            activated_items[copied_item] = value
        copied_execution_info = self.execution_info.copy(deep=True, memo=memo)
        workflow_state = self.__class__(workflow=workflow, input_values=input_values, activated_items=activated_items,
                                        values=values, output_value=deepcopy_value(value=self.output_value, memo=memo),
                                        log=self.log, execution_info=copied_execution_info, name=self.name)
        return workflow_state

    def _data_hash(self):
        workflow = hash(self.workflow)
        output = choose_hash(self.output_value)
        input_values = sum(i * choose_hash(v) for (i, v) in self.input_values.items())
        values = len(self.values) * 7
        return (workflow + output + input_values + values) % 1000000000

    def _data_eq(self, other_object: 'WorkflowState'):
        if not (self.__class__.__name__ == other_object.__class__.__name__
                and self.progress == other_object.progress
                and self.workflow == other_object.workflow
                and self.output_value == other_object.output_value):
            return False

        for index in set(list(self.input_values.keys()) + list(other_object.input_values.keys())):
            value1 = self.input_values.get(index, None)
            value2 = other_object.input_values.get(index, None)
            if value1 != value2:
                # Rechecking if input is file, in which case we tolerate different values
                if not self.workflow.inputs[index].is_file_related:
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

        for pipe, other_pipe in zip(self.workflow.pipes, other_object.workflow.pipes):
            if self.activated_items[pipe] != other_object.activated_items[other_pipe]:
                # Check pipe progress state
                return False
        return True

    @property
    def method_schemas(self):
        """ Empty schemas for WorkflowState because not directly used. """
        return {}

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None,
                **kwargs):
        """ Transform object into a dict. """
        if memo is None:
            memo = {}
        if id_memo is None:
            id_memo = {}

        if use_pointers:
            workflow_dict = self.workflow.to_dict(path=f'{path}/workflow', memo=memo)
        else:
            workflow_dict = self.workflow.to_dict(use_pointers=False)

        execution_info = self.execution_info.to_dict(block_indices={b: i for i, b in enumerate(self.workflow.blocks)})
        dict_ = self.base_dict()
        dict_.update({"execution_info": execution_info, "log": self.log, "workflow": workflow_dict})

        input_values = {}
        for input_number, value in self.input_values.items():
            if self.workflow.inputs[input_number] not in self.workflow.file_inputs:
                if use_pointers:
                    serialized_value, memo = serialize_with_pointers(value=value, memo=memo,
                                                                     path=f"{path}/input_values/{input_number}",
                                                                     id_memo=id_memo)
                else:
                    serialized_value = serialize(value)
                input_values[str(input_number)] = serialized_value

        dict_["input_values"] = input_values

        # Output value: priority for reference before values
        if self.output_value is not None:
            if use_pointers:
                serialized_output_value, memo = serialize_with_pointers(self.output_value, memo=memo,
                                                                        path=f'{path}/output_value',
                                                                        id_memo=id_memo)
            else:
                serialized_output_value = serialize(self.output_value)

            dict_.update({"output_value": serialized_output_value,
                          "output_value_type": recursive_type(self.output_value)})
        # Values
        values = {}
        for pipe, value in self.values.items():
            if not is_file_or_file_sequence(value) and pipe in self.workflow.memorized_pipes:
                pipe_index = self.workflow.pipes.index(pipe)
                if use_pointers:
                    try:
                        serialized_value, memo = serialize_with_pointers(value=value, memo=memo,
                                                                         path=f"{path}/values/{pipe_index}",
                                                                         id_memo=id_memo)
                        values[str(pipe_index)] = serialized_value
                    except SerializationError:
                        warnings.warn(f"unable to serialize {value}, dropping it from workflow state/run values",
                                      SerializationWarning)
                else:
                    values[str(pipe_index)] = serialize(value)
        dict_["values"] = values

        # In the future comment these below and rely only on activated items
        dict_["evaluated_blocks_indices"] = [i for i, b in enumerate(self.workflow.blocks)
                                             if b in self.activated_items and self.activated_items[b]]

        dict_["evaluated_pipes_indices"] = [i for i, p in enumerate(self.workflow.pipes)
                                            if p in self.activated_items and self.activated_items[p]]

        dict_["evaluated_variables_indices"] = [self.workflow.variable_indices(v) for v in self.workflow.variables
                                                if v in self.activated_items and self.activated_items[v]]
        if path == "#":
            add_references(dict_, memo, id_memo)
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, global_dict=None, pointers_memo: Dict[str, Any] = None,
                       path: str = "#", **kwargs) -> 'WorkflowState':
        """ Compute Workflow State from given dict. Handles pointers. """
        if pointers_memo is None or global_dict is None:
            global_dict, pointers_memo = update_pointers_data(global_dict=global_dict, current_dict=dict_,
                                                              pointers_memo=pointers_memo)

        workflow = Workflow.dict_to_object(dict_=dict_["workflow"], global_dict=global_dict,
                                           pointers_memo=pointers_memo, path=f"{path}/workflow")

        if "output_value" in dict_:
            output_value = deserialize(dict_["output_value"], global_dict=global_dict,
                                       pointers_memo=pointers_memo, path=f"{path}/output_value")
        else:
            output_value = None

        values = {}
        if "values" in dict_:
            for i, value in dict_["values"].items():
                values[workflow.pipes[int(i)]] = deserialize(value, global_dict=global_dict,
                                                             pointers_memo=pointers_memo, path=f"{path}/values/{i}")

        input_values = {int(i): deserialize(v, global_dict=global_dict, pointers_memo=pointers_memo,
                                            path=f"{path}/input_values/{i}") for i, v in dict_["input_values"].items()}

        activated_items = {b: i in dict_["evaluated_blocks_indices"] for i, b in enumerate(workflow.blocks)}
        activated_items.update({p: i in dict_["evaluated_pipes_indices"] for i, p in enumerate(workflow.pipes)})

        variable_indices = [tuple(i) if is_sequence(i) else i for i in dict_["evaluated_variables_indices"]]
        activated_items.update({v: workflow.variable_indices(v) in variable_indices for v in workflow.variables})

        execution_info = ExecutionInfo.dict_to_object(dict_=dict_["execution_info"],
                                                      index_to_block=dict(enumerate(workflow.blocks)))

        return cls(workflow=workflow, input_values=input_values, activated_items=activated_items, values=values,
                   output_value=output_value, log=dict_["log"], execution_info=execution_info, name=dict_["name"])

    def add_input_value(self, input_index: int, value):
        """ Add a value for given input. """
        self._activate_input(input_=self.workflow.inputs[input_index], value=value)

    def add_several_input_values(self, indices: List[int], values):
        """ Add several values for given inputs. """
        for index in indices:
            input_ = self.workflow.inputs[index]
            if index not in values:
                if self.activated_items[input_] and index in self.input_values:
                    value = self.input_values[index]
                else:
                    msg = f"Value '{input_.name}' of index '{index}' in inputs has no value."
                    if input_.type_ is not None:
                        msg += f" Should be instance of '{input_.type_}'."
                    raise ValueError(msg)
            else:
                value = values[index]
            self.add_input_value(input_index=index, value=value)

    def add_block_input_values(self, block_index: int, values):
        """ Add inputs values for given block. """
        values = {int(k): v for k, v in values.items()}  # Serialization set keys as strings
        indices = self.workflow.block_inputs_global_indices(block_index)
        self.add_several_input_values(indices=indices, values=values)

    def display_settings(self, **kwargs) -> List[DisplaySetting]:
        """ Compute the displays settings of the objects. """
        display_settings = [DisplaySetting(selector="workflow_state", type_="workflow_state", method="state_display")]

        # Displayable blocks
        display_settings.extend(self.workflow.blocks_display_settings)
        return display_settings

    def block_display(self, block_index: int, reference_path: str = "#"):
        """ Compute the display of associated block to use integrate it in the workflow run displays. """
        self.activate_inputs()
        block = self.workflow.blocks[block_index]

        selector = self.workflow.block_selector(block)
        branch = self.workflow.branch_by_display_selector[selector]
        block_args = {}
        for branch_block in branch:
            if branch_block is block:
                argpath = reference_path
            else:
                argpath = "#"
            block_args[branch_block] = {"reference_path": argpath}

        evaluated_blocks = self.evaluate_branch(blocks=branch, block_args=block_args)

        if block not in evaluated_blocks:
            msg = f"Could not reach block at index {block_index}." \
                  f"Has the workflow been run far enough to evaluate this block ?"
            raise WorkflowError(msg)
        return evaluated_blocks[block][0]  # Only one output to an Export Block

    @property
    def progress(self):
        """
        Return the current progress.

        Return a float between 0 (nothing evaluated), to 1 (every computational block evaluated).
        """
        evaluated_blocks = [self.activated_items[b] for b in self.workflow.runtime_blocks]
        progress = sum(evaluated_blocks) / len(evaluated_blocks)
        if progress == 1 and self.execution_info.end_time is None:
            self.execution_info.end_time = time.time()
        return progress

    def block_evaluation(self, block_index: int, progress_callback=lambda x: None) -> bool:
        """ Select a block to evaluate. """
        block = self.workflow.blocks[block_index]
        self.activate_inputs()
        if block in self._activable_blocks():
            nblocks = len(self.workflow.blocks)

            def block_progress_callback(x):
                progress_callback(self.progress + x / nblocks)

            self._evaluate_block(block, progress_callback=block_progress_callback)
            progress_callback(self.progress)
            return True
        return False

    def evaluate_next_block(self, progress_callback=lambda x: None) -> Optional[Block]:
        """ Evaluate a block. """
        self.activate_inputs()
        blocks = self._activable_blocks()
        if blocks:
            block_index = self.workflow.blocks.index(blocks[0])
            self.block_evaluation(block_index, progress_callback=progress_callback)
            return blocks[0]
        return None

    def continue_run(self, progress_callback=lambda x: None, export: bool = False):
        """ Evaluate all possible blocks. """
        self.activate_inputs()

        evaluated_blocks = []
        something_activated = True
        while something_activated:
            something_activated = False
            blocks = [b for b in self.workflow.runtime_blocks if b in self._activable_blocks()]
            for block in blocks:
                evaluated_blocks.append(block)
                block_index = self.workflow.blocks.index(block)
                self.block_evaluation(block_index, progress_callback=progress_callback)
                if not export:
                    progress_callback(self.progress)
                something_activated = True
        return evaluated_blocks

    def evaluate_branch(self, blocks: List[Block], block_args: Dict[Block, Any]):
        """ Evaluate all blocks of a branch, automatically finding the first executable ones. """
        self.activate_inputs()

        if not any((b in self._activable_blocks() for b in blocks)):
            raise WorkflowError("Branch cannot be evaluated because no block has all its inputs activated")

        evaluated_blocks = {}
        i = 0
        while len(evaluated_blocks) != len(blocks) and i <= len(blocks):
            next_blocks = [b for b in blocks if b in self._activable_blocks() and b not in evaluated_blocks]
            for block in next_blocks:
                kwargs = block_args[block]
                output_values = self._evaluate_block(block, **kwargs)
                evaluated_blocks[block] = output_values
            i += 1
        return evaluated_blocks

    def _activate_pipe(self, pipe: Pipe, value):
        """ Set the pipe value and activate its downstream variable. """
        self.values[pipe] = value
        self.activated_items[pipe] = True
        self._activate_variable(variable=pipe.output_variable, value=value)

    def _activate_block(self, block: Block, output_values):
        """ Activate all block outputs. """
        # Unpacking result of evaluation
        output_items = zip(block.outputs, output_values)
        for output, output_value in output_items:
            self._activate_variable(variable=output, value=output_value)
        self.activated_items[block] = True

    def _activate_variable(self, variable: Variable, value):
        """ Activate the given variable with its value and propagate activation to its outgoing pipe. """
        outgoing_pipes = self.workflow.variable_output_pipes(variable)
        if self.workflow.output == variable:
            self.output_value = value
        for outgoing_pipe in outgoing_pipes:
            self._activate_pipe(pipe=outgoing_pipe, value=value)
        self.activated_items[variable] = True

    def _activate_input(self, input_: Variable, value):  # Inputs must always be Typed
        """ Type-check, activate the variable and propagate the value to its pipe. """
        # Type checking
        value_type_check(value, input_.type_)
        input_index = self.workflow.input_index(input_)
        self.input_values[input_index] = value
        self._activate_variable(variable=input_, value=value)
        downstream_pipes = self.workflow.variable_output_pipes(input_)
        for pipe in downstream_pipes:
            self._activate_pipe(pipe=pipe, value=value)

    def _activable_blocks(self):
        """
        Returns a list of all blocks that can be activated.

        Blocks that can be activated are blocks that have all inputs ready for evaluation.
        """
        return [b for b in self.workflow.blocks if self._block_activable_by_inputs(b)
                and (not self.activated_items[b] or self.workflow.is_block_secondary(b))]

    def _block_activable_by_inputs(self, block: Block):
        """Return whether a block has all its inputs active and can be activated."""
        for function_input in block.inputs:
            if not self.activated_items[function_input]:
                return False
        return True

    def _evaluate_block(self, block, progress_callback=lambda x: x, verbose=False, **kwargs):
        """ Evaluate given block. """
        if verbose:
            log_line = f"Evaluating block {block.name}"
            self.log += log_line + '\n'
            if verbose:
                print(log_line)

        local_values = {}
        for input_ in block.inputs:
            incoming_pipe = self.workflow.variable_input_pipe(input_)
            if incoming_pipe is None:
                # Input isn't connected, it's a workflow input
                input_index = self.workflow.input_index(input_)
                value = self.input_values[input_index]
            else:
                value = self.values[incoming_pipe]
            self._activate_variable(variable=input_, value=value)
            local_values[input_] = value

        kwargs['progress_callback'] = progress_callback
        self.execution_info.before_block_memory_usage.append((block, int(psutil.Process().memory_info().vms)))
        output_values = block.evaluate(local_values, **kwargs)
        self.execution_info.after_block_memory_usage.append((block, int(psutil.Process().memory_info().vms)))
        self._activate_block(block=block, output_values=output_values)

        # Updating progress
        if progress_callback is not None:
            progress_callback(self.progress)
        return output_values

    def activate_inputs(self, check_all_inputs=False):
        """ Return whether all inputs are activated or not. """
        # Input activation
        for index, variable in enumerate(self.workflow.inputs):
            if index in self.input_values:
                self._activate_input(input_=variable, value=self.input_values[index])
            elif variable.has_default_value | variable.locked:
                self._activate_input(input_=variable, value=variable.default_value)
            elif check_all_inputs:
                msg = f"Value {variable.name} of index {index} in inputs has no value"
                if variable.type_ is not None:
                    msg += f": should be instance of {variable.type_}"
                raise ValueError(msg)

    def to_workflow_run(self, name: str = ""):
        """ Return a WorkflowRun if state is complete. """
        if self.progress == 1:
            values = {p: self.values[p] for p in self.workflow.pipes if p in self.values}
            return WorkflowRun(workflow=self.workflow, input_values=self.input_values, output_value=self.output_value,
                               values=values, activated_items=self.activated_items, log=self.log,
                               execution_info=self.execution_info, name=name)
        raise ValueError('Workflow not completed')

    def _export_formats(self):
        """ Read block to compute available export formats. """
        export_formats = DessiaObject._export_formats(self)

        # Exportable Blocks
        export_formats.extend(self.workflow.blocks_export_formats)
        return export_formats

    def export_format_from_selector(self, selector: str):
        """ Get Workflow State format from given selector. """
        for export_format in self.workflow.blocks_export_formats:
            if export_format["selector"] == selector:
                return export_format
        raise ValueError(f"No block defines an export with the selector '{selector}'")

    def export(self, stream: Union[BinaryFile, StringFile], block_index: int):
        """ Perform export. """
        block = self.workflow.blocks[block_index]
        selector = self.workflow.block_selector(block)
        branch = self.workflow.branch_by_export_format[selector]
        block_args = {b: {} for b in branch}
        evaluated_blocks = self.evaluate_branch(blocks=branch, block_args=block_args)
        if block not in evaluated_blocks:
            msg = f"Could not reach block at index {block_index}." \
                  f"Has the workflow been ran far enough to evaluate this block ?"
            raise WorkflowError(msg)
        export_stream = evaluated_blocks[block][0]  # Only one output to an Export Block
        if isinstance(stream, StringFile):
            stream.write(export_stream.getvalue())
        if isinstance(stream, BinaryFile):
            stream.write(export_stream.getbuffer())
        stream.filename = export_stream.filename
        return export_stream

    def to_markdown(self, **kwargs) -> str:
        """ Render to markdown. """
        template = dessia_common.templates.workflow_state_markdown_template
        execution_info = self.execution_info.to_markdown(self.workflow.blocks)
        return template.substitute(name=self.name, class_=self.__class__.__name__, progress=100 * self.progress,
                                   workflow_name=self.workflow.name, execution_info=execution_info)


class WorkflowRun(WorkflowState):
    """ Completed state of a workflow. """

    _allowed_methods = []

    def __init__(self, workflow: Workflow, input_values, output_value, values,
                 activated_items: Dict[Union[Pipe, Block, Variable], bool],
                 log: str = "", execution_info: ExecutionInfo = None, name: str = ""):
        filtered_values = {p: values[p] for p in workflow.memorized_pipes if p in values}
        WorkflowState.__init__(self, workflow=workflow, input_values=input_values,
                               activated_items=activated_items, values=filtered_values,
                               output_value=output_value, log=log, execution_info=execution_info, name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None,
                **kwargs):
        """ Add variable values to super WorkflowState dict. """
        if memo is None:
            memo = {}  # To make sure we have the good ref for next steps
        return WorkflowState.to_dict(self, use_pointers=use_pointers, memo=memo, path=path,
                                     id_method=id_method, id_memo=id_memo)

    def _get_from_path(self, path: str):
        """
        Extract sub-object at given path. Tries the generic function, then applies specific cases if it fails.

        Returns found object
        """
        try:
            return DessiaObject._get_from_path(self, path)
        except ExtractionError:
            segments = path.split("/")
            first_segment = segments[1]
            if first_segment == "values" and len(segments) >= 3:
                pipe_index = int(segments[2])
                pipe = self.workflow.pipes[pipe_index]
                value = self.values[pipe]
                if len(segments) > 3:
                    return DessiaObject._get_from_path(value, f"#/{'/'.join(segments[3:])}")
                return value
        raise NotImplementedError(f"WorkflowRun : Specific object from path method is not defined for path '{path}'")

    def dict_to_arguments(self, dict_: JsonSerializable, method: str):
        """ Compute run method's arguments from serialized ones. """
        if method in self._allowed_methods:
            return self.workflow.dict_to_arguments(dict_=dict_, method='run')
        raise NotImplementedError(f"Method '{method}' not in WorkflowRun allowed methods")

    def display_settings(self, **kwargs) -> List[DisplaySetting]:
        """
        Compute WorkflowRun display settings.

        Concatenate WorkflowState display_settings and inserting Workflow ones.
        """
        block_settings = self.workflow.blocks_display_settings

        displays_by_default = [s.load_by_default for s in block_settings]
        documentation = DisplaySetting(selector="Documentation", type_="markdown", method="to_markdown",
                                       load_by_default=True)
        documentation.load_by_default = not any(displays_by_default)
        return [documentation] + block_settings

    def _display_settings_from_selector(self, selector: str):
        """ Get display settings from given selector. """
        return display_settings_from_selector(display_settings=self.display_settings(), selector=selector)

    def run_again(self, input_values, progress_callback=None, name=None):
        """ Execute workflow again with given inputs. """
        return self.workflow.run(input_values=input_values, verbose=False,
                                 progress_callback=progress_callback, name=name)

    def _split_input_values(self):
        """ Split the input values for blocks and non-block variables. """
        nbv_values = [self.input_values[i] for i, v in enumerate(self.workflow.inputs) if
                      v in self.workflow.nonblock_variables]
        block_values = [self.input_values[i] for i, v in enumerate(self.workflow.inputs) if
                        v not in self.workflow.nonblock_variables]
        return nbv_values, block_values

    def _to_script(self, workflow_script):
        """ Computes elements for a to_script interpretation. """
        input_str = ""
        default_value = ""
        add_import = []

        nbv_values, block_values = self._split_input_values()

        # Blocks
        index_ = 0
        for j, block in enumerate(self.workflow.blocks):
            for i, input_ in enumerate(block.inputs):
                input_key = f"block_{j}.inputs[{i}]"
                if input_key not in workflow_script.declaration:
                    input_str += f"    workflow.input_index({input_key}): value_{j}_{i},\n"
                    default_value += generate_default_value(block_values[index_], j, i)
                    index_ += 1
                    add_import.extend(update_imports(input_.type_, input_.name))

        # NBVs
        for k, nbv in enumerate(self.workflow.nonblock_variables):
            input_str += f"    workflow.input_index(variable_{k}): value_{k},\n"
            default_value += generate_default_value(nbv_values[k], k)
            add_import.extend(update_imports(nbv.type_, nbv.name))

        return input_str, default_value, add_import

    def to_script(self):
        """ Computes a script representing the WorkflowRun. """
        workflow_script = self.workflow._to_script()
        input_str, default_value, add_import = self._to_script(workflow_script)

        workflow_script.declaration = f"{workflow_script.declaration}\n{default_value}\n\ninput_values =" \
                                      f" {{\n{input_str}}}\nworkflow_run = workflow.run(input_values=input_values)\n"
        workflow_script.imports.extend(add_import)

        return workflow_script

    def save_script_to_stream(self, stream: StringFile):
        """ Save the WorkflowRun to a python script to a stream. """
        script = self.to_script()
        stream.seek(0)
        script_imports = script.imports_to_str()
        stream.write(script_imports + "\n")
        stream.write(script.declaration + "\n")

    def save_script_to_file(self, filename: str):
        """ Save the WorkflowRun to a python script to a file on the disk. """
        if not filename.endswith('.py'):
            filename += '.py'
            print(f'Changing filename to {filename}')
        with open(filename, 'w', encoding='utf-8') as file:
            self.save_script_to_stream(file)

    def to_markdown(self, **kwargs) -> str:
        """ Render the WorkflowRun to markdown. """
        template = dessia_common.templates.workflow_run_markdown_template
        writer = MarkdownWriter(print_limit=25, table_limit=None)

        if is_sequence(self.output_value):
            output_table = f"Output is a sequence of {len(self.output_value)} elements"
        else:
            output_table = writer.object_table(self.output_value)

        execution_info = self.execution_info.to_markdown(blocks=self.workflow.blocks)
        return template.substitute(name=self.name, workflow_name=self.workflow.name,
                                   output_table=output_table, execution_info=execution_info)

    def _export_formats(self):
        """ Returns a list of export formats. """
        export_formats = super()._export_formats()
        script_export = ExportFormat(selector="py", extension="py", method_name="save_script_to_stream", text=True)
        export_formats.append(script_export)
        return export_formats

    def zip_settings(self):
        """ Returns a list of streams that contain different exports of the objects. """
        streams = []
        for export_format in self.workflow.blocks_export_formats:
            if export_format.extension != "zip":
                method_name = export_format.method_name
                stream_class = StringFile if export_format.text else BinaryFile
                stream = stream_class(filename=export_format.export_name)
                block_index = export_format.args.get('block_index')
                getattr(self, method_name)(stream, block_index)
                streams.append(stream)
        return streams


def initialize_workflow(dict_, global_dict, pointers_memo) -> Workflow:
    """ Generate blocks, pipes, detached_variables and output from a serialized state. """
    blocks = [deserialize(serialized_element=d, global_dict=global_dict, pointers_memo=pointers_memo)
              for d in dict_["blocks"]]
    if 'nonblock_variables' in dict_:
        nonblock_variables = [deserialize(serialized_element=d, global_dict=global_dict, pointers_memo=pointers_memo)
                              for d in dict_['nonblock_variables']]
    else:
        nonblock_variables = []

    connected_nbvs = {v: False for v in nonblock_variables}
    pipes = deserialize_pipes(pipes_dict=dict_['pipes'], blocks=blocks, nonblock_variables=nonblock_variables,
                              connected_nbvs=connected_nbvs)

    if dict_['output'] is not None:
        output = blocks[dict_['output'][0]].outputs[dict_['output'][2]]
    else:
        output = None
    detached_variables = [v for v, is_connected in connected_nbvs.items() if not is_connected]
    return Workflow(blocks=blocks, pipes=pipes, output=output, detached_variables=detached_variables)


def deserialize_pipes(pipes_dict, blocks, nonblock_variables, connected_nbvs):
    """ Generate all pipes from a dict. """
    pipes = []
    for source, target in pipes_dict:
        if isinstance(source, int):
            variable1 = nonblock_variables[source]
            connected_nbvs[variable1] = True
        else:
            ib1, _, ip1 = source
            variable1 = blocks[ib1].outputs[ip1]

        if isinstance(target, int):
            variable2 = nonblock_variables[target]
            connected_nbvs[variable2] = True
        else:
            ib2, _, ip2 = target
            variable2 = blocks[ib2].inputs[ip2]

        if connected_nbvs.get(variable1, False):
            # TODO This is an horrible hotfix to prevent NBVs type to be wrongly deserialized as strings.
            # We MUST refactor non_block_variables to allow them to be detached and remove detached variables concept.
            # The detached variable concept is completely flawed. NBVs MUST be able to exist without downstream blocks
            # This is plainly wrong because a NBV could be linked to several input ports with concurrent types
            # (by inheritance for example). Here we are resetting the NBV type to whatever we find last.
            # All in all, as blocks reset their inputs every time (trust user code instead of frontend json),
            # this is a way to reset NBVs type to user-code value and avoiding deserialization as string,
            # which are two completely different issues.
            variable1.type_ = variable2.type_
        pipes.append(Pipe(variable1, variable2))
    return pipes


def value_type_check(value, type_):
    """
    Type propagation.

    Check if the value as the specified type.
    """
    if type_ is None:
        return False
    try:  # TODO: Sub-scripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass
    return True
