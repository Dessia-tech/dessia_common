#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gathers all workflow relative features.
"""
import ast
import time
import datetime
import tempfile
import json
import webbrowser

import io
from typing import List, Union, Type, Any, Dict, Tuple, Optional
from copy import deepcopy
import warnings
import traceback as tb
import networkx as nx

import dessia_common.errors
from dessia_common.graph import get_column_by_node
from dessia_common.templates import workflow_template
from dessia_common.core import DessiaObject, is_sequence, JSONSCHEMA_HEADER, jsonschema_from_annotation, \
    deserialize_argument, set_default_value, DisplaySetting

from dessia_common.utils.types import serialize_typing, deserialize_typing, recursive_type, typematch
from dessia_common.utils.copy import deepcopy_value
from dessia_common.utils.docstrings import FAILED_ATTRIBUTE_PARSING, EMPTY_PARSED_ATTRIBUTE
from dessia_common.utils.diff import choose_hash
from dessia_common.utils.helpers import prettyname

from dessia_common.typings import JsonSerializable, MethodType
from dessia_common.files import StringFile, BinaryFile
from dessia_common.displays import DisplayObject
from dessia_common.breakdown import attrmethod_getter, ExtractionError
from dessia_common.errors import SerializationError
from dessia_common.warnings import SerializationWarning
from dessia_common.exports import ExportFormat
from dessia_common.serialization import deserialize, serialize_with_pointers, serialize, update_pointers_data, \
    serialize_dict, is_serializable

from dessia_common.workflow.utils import ToScriptElement


class Variable(DessiaObject):
    """ Variable for workflow. """
    _standalone_in_db = False
    _eq_is_data_eq = False
    has_default_value: bool = False

    def __init__(self, name: str = '', position=None):
        DessiaObject.__init__(self, name=name)
        if position is None:
            self.position = (0, 0)
        else:
            self.position = position

    def to_dict(self, use_pointers=True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Serialize the variable with custom logic. """
        dict_ = DessiaObject.base_dict(self)
        dict_.update({'has_default_value': self.has_default_value, 'position': self.position})
        return dict_

    def _to_script(self) -> ToScriptElement:
        script = self._get_to_script_elements()
        script.declaration = f"{self.__class__.__name__}({script.declaration})"

        script.imports.append(self.full_classname)
        return script

    def _get_to_script_elements(self):
        declaration = f"name='{self.name}', position={self.position}"
        return ToScriptElement(declaration=declaration, imports=[], imports_as_is=[])


class TypedVariable(Variable):
    """ Variable for workflow with a typing. """
    has_default_value: bool = False

    def __init__(self, type_: Type, name: str = '', position=None):
        Variable.__init__(self, name=name, position=position)
        self.type_ = type_

    def to_dict(self, use_pointers=True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Serializes the object with specific logic. """
        dict_ = super().to_dict(use_pointers, memo, path)
        dict_.update({'type_': serialize_typing(self.type_)})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'TypedVariable':
        """
        Compute variable from dict.

        TODO Remove this ?
        """
        type_ = deserialize_typing(dict_['type_'])
        return cls(type_=type_, name=dict_['name'], position=dict_.get("position"))

    def copy(self, deep: bool = False, memo=None):
        """ Copies and gives a new object with no linked data. """
        return TypedVariable(type_=self.type_, name=self.name)

    def _get_to_script_elements(self) -> ToScriptElement:
        script = super()._get_to_script_elements()

        script.declaration += f", type_={self.type_.__name__}"

        if "builtins" not in serialize_typing(self.type_):
            script.imports.append(serialize_typing(self.type_))
        return script


class VariableWithDefaultValue(Variable):
    """
    A variable with a default value.

    TODO Isn't this always typed ?
    """
    has_default_value: bool = True

    def __init__(self, default_value: Any, name: str = '', position=None):
        Variable.__init__(self, name=name, position=position)
        self.default_value = default_value


class TypedVariableWithDefaultValue(TypedVariable):
    """
    Workflow variables wit a type and a default value.

    TODO Can this be just VariableWithDefaultValue ? Type is induced ?
    """
    has_default_value: bool = True

    def __init__(self, type_: Type, default_value: Any, name: str = '', position=None):
        TypedVariable.__init__(self, type_=type_, name=name, position=position)
        self.default_value = default_value

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Serialize the variable with custom logic. """
        dict_ = super().to_dict(use_pointers, memo, path)
        dict_.update({'default_value': serialize(self.default_value)})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'TypedVariableWithDefaultValue':
        """
        Compute variable from dict.

        TODO Remove this ?
        """
        type_ = deserialize_typing(dict_['type_'])
        default_value = deserialize(dict_['default_value'], global_dict=global_dict, pointers_memo=pointers_memo)
        return cls(type_=type_, default_value=default_value, name=dict_['name'], position=dict_.get('position'))

    def copy(self, deep: bool = False, memo=None):
        """
        Copy a TypedVariableWithDefaultValue.

        :param deep: DESCRIPTION, defaults to False
        :type deep: bool, optional

        :param memo: a memo to use, defaults to None
        :type memo: TYPE, optional

        :return: The copied object
        """
        if memo is None:
            memo = {}
        copied_default_value = deepcopy_value(self.default_value, memo=memo)
        return TypedVariableWithDefaultValue(type_=self.type_, default_value=copied_default_value, name=self.name)

    def _to_script(self) -> ToScriptElement:
        warnings.warn("to_script method is not implemented for TypedVariableWithDefaultValue yet. "
                      "We are losing the default value as we call the TypedVariable method")
        casted_variable = TypedVariable(type_=self.type_, name=self.name, position=self.position)
        return casted_variable._to_script()


NAME_VARIABLE = TypedVariable(type_=str, name="Result Name")


def set_block_variable_names_from_dict(func):
    """ Inspect func arguments to compute black variable names. """
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
    """ An Abstract block. Do not instantiate alone. """
    _standalone_in_db = False
    _eq_is_data_eq = False
    _non_serializable_attributes = []

    def __init__(self, inputs: List[Variable], outputs: List[Variable],
                 position: Tuple[float, float] = None, name: str = ''):
        if position is None:
            self.position = (0, 0)
        else:
            self.position = position
        self.inputs = inputs
        self.outputs = outputs
        DessiaObject.__init__(self, name=name)

    def equivalent_hash(self):
        """
        Custom hash of block that doesn't overwrite __hash__ as we do not want to lose python default equality behavior.

        Used by workflow module only.
        """
        return len(self.__class__.__name__)

    def equivalent(self, other):
        """
        Custom eq of block that does not overwrite __eq__ as we do not want to lose python default equality behavior.

        Used by workflow module only.
        """
        return self.__class__.__name__ == other.__class__.__name__

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Serialize the block with custom logic. """
        dict_ = DessiaObject.base_dict(self)
        dict_['inputs'] = [i.to_dict() for i in self.inputs]
        dict_['outputs'] = [o.to_dict() for o in self.outputs]
        if self.position is not None:
            dict_['position'] = list(self.position)
        else:
            dict_['position'] = self.position
        return dict_

    def jointjs_data(self):
        """ Deprecated HTML computation. """
        data = {'block_class': self.__class__.__name__}
        if self.name != '':
            data['name'] = self.name
        else:
            data['name'] = self.__class__.__name__
        return data

    def _docstring(self):
        """ Base function for submodel docstring computing. """
        block_docstring = {i: EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring

    def base_script(self) -> str:
        """ Generate a chunk of script that denotes the arguments of a base block. """
        return f"name='{self.name}', position={self.position}"

    def is_valid(self, level: str = 'error') -> bool:  # TODO: Change this in further releases
        """ Always return True for now. """
        return True


class Pipe(DessiaObject):
    """
    Bind two variables of a Workflow.

    :param input_variable: The input varaible of the pipe correspond to the start of the arrow, its tail.
    :type input_variable: Variable
    :param output_variable: The output variable of the pipe correpond to the end of the arrow, its hat.
    :type output_variable: Variable
    """
    _eq_is_data_eq = False

    def __init__(self, input_variable: Variable, output_variable: Variable, name: str = ''):
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.memorize = False
        DessiaObject.__init__(self, name=name)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Transform the pipe into a dict. """
        return {'input_variable': self.input_variable, 'output_variable': self.output_variable,
                'memorize': self.memorize}


class WorkflowError(Exception):
    """ Specific WorkflowError Exception. """


class Workflow(Block):
    """
    Class Block of Workflows.

    :param blocks:
        A List with all the Blocks used by the Worklow.
    :type blocks: List[Block]

    :param pipes:
        A List of Pipe objects.
    :type pipes: List[Pipe]

    :param imposed_variable_values:
        A dictionary of imposed variable values.
    :type imposed_variable_values: Dict

    :param description:
        A short description that will be displayed on workflow card (frontend). Should be shorter than 100 chars
    :type description: str

    :param documentation:
        A long documentation that will be displayed on workflow page (frontend). Can use markdown elements.
    :type documentation: str

    :param name:
        The name of the workflow.
    :type name: str
    """
    _standalone_in_db = True
    _allowed_methods = ['run', 'start_run']
    _eq_is_data_eq = True
    _jsonschema = {
        "definitions": {}, "$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "title": "Workflow",
        "required": ["blocks", "pipes", "outputs"], "python_typing": 'dessia_common.workflow.core.Workflow',
        "classes": ["dessia_common.workflow.core.Workflow"], "standalone_in_db": True,
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
                "type": "array", "order": 2, "python_typing": "List[dessia_common.workflow.Variable]",
                'items': {
                    'type': 'array', 'items': {'type': 'number'},
                    'python_typing': "dessia_common.workflow.Variable"
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
                 detached_variables: List[TypedVariable] = None, description: str = "",
                 documentation: str = "", name: str = ""):
        self.blocks = blocks
        self.pipes = pipes

        if imposed_variable_values is None:
            imposed_variable_values = {}
        self.imposed_variable_values = imposed_variable_values

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

        self._utd_graph = False

        inputs = [v for v in self.variables if v not in self.imposed_variable_values
                  and len(nx.ancestors(self.graph, v)) == 0]
        # if not hasattr(variable, 'type_'):
        #     raise WorkflowError('Workflow as an untyped input variable: {}'.format(variable.name))

        self.description = description
        self.documentation = documentation

        outputs = []
        self.output = output
        if output is not None:
            outputs.append(output)
            found_output = False
            i = 0
            while not found_output and i < len(self.blocks):
                found_output = output in self.blocks[i].outputs
                i += 1
            if not found_output:
                raise WorkflowError("workflow's output is not in any block's outputs")

        found_name = False
        i = 0
        all_nbvs = self.nonblock_variables + self.detached_variables
        while not found_name and i < len(all_nbvs):
            variable = all_nbvs[i]
            found_name = variable.name == "Result Name"
            i += 1
        if not found_name:
            self.detached_variables.insert(0, NAME_VARIABLE)

        Block.__init__(self, inputs=inputs, outputs=outputs, name=name)

        self.block_selectors = {}
        for i, block in enumerate(self.blocks):
            name = block.name
            if not name:
                if block in self.display_blocks:
                    name = block.type_
                elif block in self.export_blocks:
                    name = block.extension
                else:
                    name = "Block"
            self.block_selectors[block] = f"{name} ({i})"

        self.branch_by_display_selector = self.branch_by_selector(self.display_blocks)
        self.branch_by_export_format = self.branch_by_selector(self.export_blocks)

    @classmethod
    def generate_empty(cls):
        """ Generate an empty workflow (mostly used by frontend to compute an init dict). """
        return cls(blocks=[], pipes=[], output=None)

    @property
    def nodes(self):
        """ Return the list of blocks and nonblock_variables (nodes) of the Workflow. """
        return self.blocks + self.nonblock_variables

    def handle_pipe(self, pipe):
        """
        Perform some initialization action on a pipe and its variables.
        """
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
        """
        Perform some initialization action on a block and its variables.
        """
        if isinstance(block, Workflow):
            raise ValueError("Using workflow as blocks is forbidden, use WorkflowBlock wrapper instead")
        self.variables.extend(block.inputs)
        self.variables.extend(block.outputs)
        try:
            self.coordinates[block] = (0, 0)
        except ValueError as err:
            raise ValueError(f"Cannot serialize block {block} ({block.name})") from err

    def _data_hash(self):
        output_hash = hash(self.variable_indices(self.output))
        base_hash = len(self.blocks) + 11 * len(self.pipes) + 23 * len(self.imposed_variable_values) + output_hash
        block_hash = int(sum(b.equivalent_hash() for b in self.blocks) % 1e6)
        return (base_hash + block_hash) % 1000000000

    def _data_eq(self, other_object) -> bool:
        if hash(self) != hash(other_object) or not Block.equivalent(self, other_object):
            return False

        # TODO: temp , reuse graph to handle block order!!!!
        for block1, block2 in zip(self.blocks, other_object.blocks):
            if not block1.equivalent(block2):
                return False

        if not self._equivalent_pipes(other_object):
            return False

        if not self._equivalent_imposed_variables_values(other_object):
            return False
        return True

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

    def _equivalent_imposed_variables_values(self, other_wf) -> bool:
        ivvs = set()
        other_ivvs = set()
        for imposed_key, other_imposed_key in zip(self.imposed_variable_values.keys(),
                                                  other_wf.imposed_variable_values.keys()):
            variable_index = self.variable_index(imposed_key)
            ivvs.add((variable_index, self.imposed_variable_values[imposed_key]))

            other_variable_index = other_wf.variable_index(other_imposed_key)
            other_ivvs.add((other_variable_index, other_wf.imposed_variable_values[other_imposed_key]))

        return ivvs == other_ivvs

    def __deepcopy__(self, memo=None):
        """ Return the deep copy. """
        if memo is None:
            memo = {}

        blocks = [b.__deepcopy__() for b in self.blocks]
        output_adress = self.variable_indices(self.output)
        if output_adress is None:
            output = None
        else:
            output_block = blocks[output_adress[0]]
            output = output_block.outputs[output_adress[2]]

        copied_workflow = Workflow(blocks=blocks, pipes=[], output=output, name=self.name)

        pipes = self.copy_pipes(copied_workflow)

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            new_variable = copied_workflow.variable_from_index(self.variable_indices(variable))
            imposed_variable_values[new_variable] = value

        copied_workflow = Workflow(blocks=blocks, pipes=pipes, output=output,
                                   imposed_variable_values=imposed_variable_values, name=self.name)
        return copied_workflow

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

    def branch_by_selector(self, blocks: List[Block]):
        """
        Return the corresponding branch to each display or export selector.
        """
        selector_branches = {}
        for block in blocks:
            branch = self.secondary_branch_blocks(block)
            selector = self.block_selectors[block]
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
            block_index = self.blocks.index(block)
            settings = block._display_settings(block_index)
            if settings is not None:
                settings.selector = self.block_selectors[block]
                display_settings.append(settings)
        return display_settings

    @staticmethod
    def display_settings() -> List[DisplaySetting]:
        """ Compute the displays settings of the workflow. """
        display_settings = [DisplaySetting('documentation', 'markdown', 'to_markdown', None),
                            DisplaySetting('workflow', 'workflow', 'to_dict', None)]
        return display_settings

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
                format_.selector = self.block_selectors[block]
                export_formats.append(format_)
        return export_formats

    def _export_formats(self):
        """ Read block to compute available export formats. """
        export_formats = DessiaObject._export_formats(self)
        script_export = ExportFormat(selector="py", extension="py", method_name="save_script_to_stream", text=True)
        export_formats.append(script_export)
        return export_formats

    def to_markdown(self):
        """ Set workflow documentation as markdown. """
        return self.documentation

    def _docstring(self):
        """ Compute documentation of all blocks. """
        return [b._docstring() for b in self.blocks]

    @property
    def _method_jsonschemas(self):
        """ Compute the run jsonschema (had to be overloaded). """
        jsonschemas = {'run': deepcopy(JSONSCHEMA_HEADER)}
        jsonschemas['run'].update({'classes': ['dessia_common.workflow.Workflow']})
        properties_dict = jsonschemas['run']['properties']
        required_inputs = []
        parsed_attributes = {}
        for i, input_ in enumerate(self.inputs + self.detached_variables):
            current_dict = {}
            if isinstance(input_, TypedVariable) or isinstance(input_, TypedVariableWithDefaultValue):
                annotation = (str(i), input_.type_)
            else:
                annotation = (str(i), Any)
            if input_ in self.nonblock_variables or input_ in self.detached_variables:
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
            if input_ not in self.imposed_variable_values:  # Removes from Optional in edits
                properties_dict[str(i)] = current_dict[str(i)]

        jsonschemas['run'].update({'required': required_inputs, 'method': True,
                                   'python_typing': serialize_typing(MethodType)})
        jsonschemas['start_run'] = deepcopy(jsonschemas['run'])
        jsonschemas['start_run']['required'] = []
        return jsonschemas

    def to_dict(self, use_pointers=True, memo=None, path='#', id_method=True, id_memo=None):
        """
        Compute a dict from the object content.
        """
        if memo is None:
            memo = {}

        # self.refresh_blocks_positions()
        dict_ = Block.to_dict(self)
        dict_['object_class'] = 'dessia_common.workflow.core.Workflow'  # TO force migrating from dessia_common.workflow
        blocks = [b.to_dict() for b in self.blocks]

        pipes = [self.pipe_variable_indices(p) for p in self.pipes]

        output = self.variable_indices(self.output)
        dict_.update({'blocks': blocks, 'pipes': pipes, 'output': output,
                      'nonblock_variables': [v.to_dict() for v in self.nonblock_variables + self.detached_variables],
                      'package_mix': self.package_mix()})

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            var_index = self.variable_indices(variable)

            if use_pointers:
                ser_value, memo = serialize_with_pointers(value=value, memo=memo,
                                                          path=f"{path}/imposed_variable_values/{var_index}")
            else:
                ser_value = serialize(value)
            imposed_variable_values[str(var_index)] = ser_value

        dict_.update({'description': self.description, 'documentation': self.documentation,
                      'imposed_variable_values': imposed_variable_values})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Workflow':
        """
        Recompute the object from a dict.
        """
        if pointers_memo is None or global_dict is None:
            global_dict, pointers_memo = update_pointers_data(global_dict=global_dict, current_dict=dict_,
                                                              pointers_memo=pointers_memo)

        workflow = initialize_workflow(dict_=dict_, global_dict=global_dict, pointers_memo=pointers_memo)

        if 'imposed_variable_values' in dict_ and 'imposed_variables' in dict_:
            # Legacy support of double list
            imposed_variable_values = {}
            for variable_index, serialized_value in zip(dict_['imposed_variables'], dict_['imposed_variable_values']):
                value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                variable = workflow.variable_from_index(variable_index)
                imposed_variable_values[variable] = value
        else:
            imposed_variable_values = {}
            if 'imposed_variable_indices' in dict_:
                for variable_index in dict_['imposed_variable_indices']:
                    variable = workflow.variable_from_index(variable_index)
                    imposed_variable_values[variable] = variable.default_value
            if 'imposed_variable_values' in dict_:
                # New format with a dict
                for variable_index_str, serialized_value in dict_['imposed_variable_values'].items():
                    variable_index = ast.literal_eval(variable_index_str)
                    value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                    variable = workflow.variable_from_index(variable_index)
                    imposed_variable_values[variable] = value

            if 'imposed_variable_indices' not in dict_ and 'imposed_variable_values' not in dict_:
                imposed_variable_values = None

        description = dict_.get("description", "")
        documentation = dict_.get("documentation", "")

        return cls(blocks=workflow.blocks, pipes=workflow.pipes, output=workflow.output,
                   imposed_variable_values=imposed_variable_values, description=description,
                   documentation=documentation, name=dict_["name"])

    def dict_to_arguments(self, dict_: JsonSerializable, method: str, global_dict=None,
                          pointers_memo=None, path='#'):
        """
        Process a json of arguments and deserialize them.
        """
        dict_ = {int(k): v for k, v in dict_.items()}  # serialisation set keys as strings
        if method in self._allowed_methods:
            name = None
            arguments_values = {}
            for i, input_ in enumerate(self.inputs):
                has_default = input_.has_default_value
                if not has_default or (has_default and i in dict_):
                    value = dict_[i]
                    path_value = f'{path}/inputs/{i}'
                    deserialized_value = deserialize_argument(type_=input_.type_, argument=value,
                                                              global_dict=global_dict,
                                                              pointers_memo=pointers_memo, path=path_value)
                    if input_.name == "Result Name":
                        name = deserialize_argument(type_=input_.type_, argument=value, global_dict=global_dict,
                                                    pointers_memo=pointers_memo, path=path_value)
                    arguments_values[i] = deserialized_value
            if name is None and len(self.inputs) in dict_ and isinstance(dict_[len(self.inputs)], str):
                # Hot fixing name not attached
                name = dict_[len(self.inputs)]
            return {'input_values': arguments_values, 'name': name}
        raise NotImplementedError(f"Method {method} not in Workflow allowed methods")

    def _run_dict(self) -> Dict:
        dict_ = {}

        copied_ivv = {}
        for variable, value in self.imposed_variable_values.items():
            variable_index = self.variables.index(variable)
            copied_ivv[variable_index] = value

        cached_ivv = self.imposed_variable_values
        self.imposed_variable_values = {}
        copied_workflow = self.copy()
        self.imposed_variable_values = cached_ivv
        # We need to clear the imposed_variables_values and then copy the workflow in order
        # to have the good input indices in the loop bellow

        for input_index, input_ in enumerate(copied_workflow.inputs):
            variable_index = copied_workflow.variables.index(input_)
            if variable_index in copied_ivv.keys():
                dict_[input_index] = serialize(copied_ivv[variable_index])
            elif isinstance(input_, TypedVariableWithDefaultValue):
                dict_[input_index] = serialize(input_.default_value)

        return dict_

    def _start_run_dict(self) -> Dict:
        return {}

    def method_dict(self, method_name: str = None, method_jsonschema: Any = None) -> Dict:
        """ Wrapper method to get dicts of run and start_run methods. """
        if method_name == 'run':
            return self._run_dict()
        if method_name == 'start_run':
            return self._start_run_dict()
        raise WorkflowError(f"Calling method_dict with unknown method_name '{method_name}'")

    def variable_from_index(self, index: Union[int, Tuple[int, int, int]]):
        """
        Index elements are, in order : (Block index : int, Port side (0: input, 1: output), Port index : int).
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
        """
        Cached property for graph.
        """
        if not self._utd_graph:
            self._cached_graph = self._graph()
            self._utd_graph = True
        return self._cached_graph

    graph = property(_get_graph)

    def _graph(self):
        """
        Compute the networkx graph of the workflow.
        """
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
        Return blocks that are upstream for output.
        """
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

    def secondary_branch_blocks(self, block: Block) -> List[Block]:
        """
        Compute the necessary upstreams blocks to run a part of a workflow that leads to the given block.

        It stops looking for blocks when it reaches the main branch, and memorize the connected pipe

        :param block: Block that is the target of the secondary branch
        :type block: Block
        """
        upstream_blocks = self.upstream_blocks(block)
        branch_blocks = [block]
        i = 0
        candidates = upstream_blocks
        while candidates and i <= len(self.blocks):
            candidates = []
            for upstream_block in upstream_blocks:
                if upstream_block not in self.runtime_blocks and upstream_block not in branch_blocks:
                    branch_blocks.insert(0, upstream_block)
                    candidates.extend(self.upstream_blocks(upstream_block))
            upstream_blocks = candidates
            i += 1
        for branch_block in branch_blocks:
            upstream_blocks = self.upstream_blocks(branch_block)
            for upstream_block in upstream_blocks:
                if upstream_block in self.runtime_blocks:
                    for pipe in self.pipes_between_blocks(upstream_block, branch_block):
                        pipe.memorize = True
        return branch_blocks

    def pipe_from_variable_indices(self, upstream_indices: Union[int, Tuple[int, int, int]],
                                   downstream_indices: Union[int, Tuple[int, int, int]]) -> Pipe:
        """Get a pipe from the global indices of its attached variables."""
        for pipe in self.pipes:
            if self.variable_indices(pipe.input_variable) == upstream_indices \
                    and self.variable_indices(pipe.output_variable) == downstream_indices:
                return pipe
        msg = f"No pipe has {upstream_indices} as upstream variable and {downstream_indices} as downstream variable"
        raise ValueError(msg)

    def pipe_variable_indices(self, pipe: Pipe) -> Tuple[Union[int, Tuple[int, int, int]],
                                                         Union[int, Tuple[int, int, int]]]:
        """Return the global indices of a pipe's attached variables."""
        return self.variable_indices(pipe.input_variable), self.variable_indices(pipe.output_variable)

    def variable_input_pipe(self, variable: Variable) -> Optional[Pipe]:
        """Get the incoming pipe for a variable. If variable is not connected, returns None."""
        incoming_pipes = [p for p in self.pipes if p.output_variable == variable]
        if incoming_pipes:  # Inputs can only be connected to one pipe
            incoming_pipe = incoming_pipes[0]
            return incoming_pipe
        return None

    def variable_output_pipes(self, variable: Variable) -> List[Optional[Pipe]]:
        """Compute all pipes going out a given variable."""
        return [p for p in self.pipes if p.input_variable == variable]

    def pipes_between_blocks(self, upstream_block: Block, downstream_block: Block):
        """
        Compute all the pipes linking two blocks.
        """
        pipes = []
        for outgoing_pipe in self.block_outgoing_pipes(upstream_block):
            if outgoing_pipe is not None and outgoing_pipe in self.block_incoming_pipes(downstream_block):
                pipes.append(outgoing_pipe)
        return pipes

    def block_incoming_pipes(self, block: Block) -> List[Optional[Pipe]]:
        """Get incoming pipes for every block variable."""
        return [self.variable_input_pipe(i) for i in block.inputs]

    def block_outgoing_pipes(self, block: Block) -> List[Pipe]:
        """
        Return all block outgoing pipes.
        """
        outgoing_pipes = []
        for output in block.outputs:
            outgoing_pipes.extend(self.variable_output_pipes(output))
        return outgoing_pipes

    def upstream_blocks(self, block: Block) -> List[Block]:
        """
        Return a list of given block's upstream blocks.
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

    def get_upstream_nbv(self, variable: Variable) -> Variable:
        """
        If given variable has an upstream nonblock_variable, return it otherwise return given variable itself.
        """
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

    def variable_indices(self, variable: Variable) -> Optional[Union[Tuple[int, int, int], int]]:
        """
        Return global adress of given variable as a tuple or an int.

        If variable is non block, return index of variable in variables sequence
        Else returns global adress (ib, i, ip)
        """
        if variable is None:
            return None

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

    def index(self, variable):
        """
        Deprecated, will be remove in version 0.8.0.
        """
        warnings.warn("index method is deprecated, use input_index instead", DeprecationWarning)
        return self.input_index(variable)

    def input_index(self, variable: Variable) -> Optional[int]:
        """
        If variable is a workflow input, returns its index.
        """
        upstream_variable = self.get_upstream_nbv(variable)
        if upstream_variable in self.inputs:
            return self.inputs.index(upstream_variable)
        return None

    def variable_index(self, variable: Variable) -> int:
        """
        Return variable index in variables sequence.
        """
        return self.variables.index(variable)

    def block_inputs_global_indices(self, block_index: int) -> List[int]:
        """
        Return given block inputs global indices in inputs sequence.
        """
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
            if isinstance(variable, TypedVariable):
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
        if variable == other_variable:
            # If this is the same variable, it is not compatible
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

        if not (isinstance(variable, TypedVariable) and isinstance(other_variable, TypedVariable)):
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

        :returns: list[ColumnLayout] where ColumnLayout is list[node_index]
        """
        column_by_node = get_column_by_node(graph)
        nodes_by_column = {}
        for node, column_index in column_by_node.items():
            node_index = self.nodes.index(node)
            nodes_by_column[column_index] = nodes_by_column.get(column_index, []) + [node_index]

        return [column_list for column_list in nodes_by_column.values()]

    def layout(self):
        """
        Stores a workflow gaph layout.

        :returns: list[GraphLayout] where GraphLayout is list[ColumnLayout] and ColumnLayout is list[node_index]
        """
        digraph = self.layout_graph
        graph = digraph.to_undirected()
        connected_components = nx.connected_components(graph)

        return [self.graph_columns(digraph.subgraph(cc)) for cc in list(connected_components)]

    def plot_graph(self):
        """
        Plot graph by means of networking and matplotlib.
        """
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
        Full run of a workflow. Yields a WorkflowRun.
        """
        log = ''

        state = self.start_run(input_values)
        state.activate_inputs(check_all_inputs=True)

        start_time = time.time()
        start_timestamp = datetime.datetime.now()

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

        if not name:
            timestamp = start_timestamp.strftime("%m-%d (%H:%M)")
            name = f"{self.name} @ [{timestamp}]"
        return state.to_workflow_run(name=name)

    def start_run(self, input_values=None, name: str = None):
        """Partial run of a workflow. Yields a WorkflowState."""
        return WorkflowState(self, input_values=input_values, name=name)

    def jointjs_layout(self, min_horizontal_spacing=300, min_vertical_spacing=200, max_height=800, max_length=1500):
        """
        Deprecated workflow layout. Used only in jointjs_data method.
        """
        coordinates = {}
        elements_by_distance = {}
        if self.output:
            for element in self.nodes:
                distances = []
                paths = nx.all_simple_paths(self.graph, element, self.output)
                for path in paths:
                    distance = 1
                    for path_element in path[1:-1]:
                        if path_element in self.blocks + self.nonblock_variables:
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

    def jointjs_data(self):
        """
        Compute the data needed for jointjs ploting.
        """
        coordinates = self.jointjs_layout()
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
            if self.is_variable_nbv(pipe.input_variable):
                node1 = input_index
            else:
                ib1, is1, ip1 = input_index
                if is1:
                    block = self.blocks[ib1]
                    ip1 += len(block.inputs)

                node1 = [ib1, ip1]

            output_index = self.variable_indices(pipe.output_variable)
            if self.is_variable_nbv(pipe.output_variable):
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

    def plot(self, **kwargs):
        """
        Display workflow in web browser.
        """
        data = json.dumps(self.jointjs_data())
        rendered_template = workflow_template.substitute(workflow_data=data)

        temp_file = tempfile.mkstemp(suffix='.html')[1]
        with open(temp_file, 'wb') as file:
            file.write(rendered_template.encode('utf-8'))
        webbrowser.open('file://' + temp_file)

    def is_valid(self, level='error'):
        """
        Tell if the workflow is valid by checking type compatibility of pipes inputs/outputs.
        """
        for pipe in self.pipes:
            if hasattr(pipe.input_variable, 'type_') and hasattr(pipe.output_variable, 'type_'):
                type1 = pipe.input_variable.type_
                type2 = pipe.output_variable.type_
                if type1 != type2:
                    try:
                        issubclass(pipe.input_variable.type_, pipe.output_variable.type_)
                    except TypeError as error:  # TODO: need of a real typing check
                        consistent = True
                        if not consistent:
                            raise TypeError(f"Inconsistent pipe type from pipe input {pipe.input_variable.name}"
                                            f"to pipe output {pipe.output_variable.name}: "
                                            f"{pipe.input_variable.type_} incompatible with"
                                            f"{pipe.output_variable.type_}") from error
        return True

    def package_mix(self) -> Dict[str, float]:
        """
        Compute a structure showing percentages of packages used.
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

    def _to_script(self, prefix: str = '') -> ToScriptElement:
        """
        Computes elements for a to_script interpretation.

        :returns: ToSriptElement
        """
        workflow_output_index = self.variable_indices(self.output)
        if workflow_output_index is None:
            raise ValueError("A workflow output must be set")

        # --- Blocks ---
        blocks_str = ""
        imports = []
        imports_as_is = []
        for iblock, block in enumerate(self.blocks):
            block_script = block._to_script(prefix)
            imports.extend(block_script.imports)
            if block_script.before_declaration is not None:
                blocks_str += f"{block_script.before_declaration}\n"
            blocks_str += f'{prefix}block_{iblock} = {block_script.declaration}\n'
        blocks_str += f"{prefix}blocks = [{', '.join([prefix + 'block_' + str(i) for i in range(len(self.blocks))])}]\n"

        # --- NBVs ---
        nbvs_str = ""
        for nbv_index, nbv in enumerate(self.nonblock_variables):
            nbv_script = nbv._to_script()
            imports.extend(nbv_script.imports)
            imports_as_is.extend(nbv_script.imports_as_is)
            nbvs_str += f"{prefix}variable_{nbv_index} = {nbv_script.declaration}\n"

        # --- Pipes ---
        if len(self.pipes) > 0:
            imports.append(self.pipes[0].full_classname)

        pipes_str = ""
        for ipipe, pipe in enumerate(self.pipes):
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
            pipes_str += f"{prefix}pipe_{ipipe} = Pipe({input_name}, {output_name})\n"
        pipes_str += f"{prefix}pipes = [{', '.join([prefix + 'pipe_' + str(i) for i in range(len(self.pipes))])}]\n"

        # --- Building script ---
        output_name = f"{prefix}block_{workflow_output_index[0]}.outputs[{workflow_output_index[2]}]"

        full_script = f"{blocks_str}\n" \
                      f"{nbvs_str}\n" \
                      f"{pipes_str}\n" \
                      f"{prefix}workflow = " \
                      f"Workflow({prefix}blocks, {prefix}pipes, output={output_name}, name='{self.name}')\n"

        for key, value in self.imposed_variable_values.items():
            variable_indice = self.variable_indices(key)
            if self.is_variable_nbv(key):
                variable_str = variable_indice
            else:
                [block_index, _, variable_index] = variable_indice
                variable_str = f"{prefix}blocks[{block_index}].inputs[{variable_index}]"
            full_script += f"{prefix}workflow.imposed_variable_values[{variable_str}] = {value}\n"
        return ToScriptElement(declaration=full_script, imports=imports, imports_as_is=imports_as_is)

    def to_script(self) -> str:
        """
        Compute a script representing the workflow.
        """
        workflow_output_index = self.variable_indices(self.output)
        if workflow_output_index is None:
            raise ValueError("A workflow output must be set")

        self_script = self._to_script()
        self_script.imports.append(self.full_classname)

        script_imports = self_script.imports_to_str()

        return f"{script_imports}\n" \
               f"{self_script.declaration}"

    def save_script_to_stream(self, stream: io.StringIO):
        """
        Save the workflow to a python script to a stream.
        """
        string = self.to_script()
        stream.seek(0)
        stream.write(string)

    def save_script_to_file(self, filename: str):
        """
        Save the workflow to a python script to a file on the disk.
        """
        if not filename.endswith('.py'):
            filename += '.py'
        with open(filename, 'w', encoding='utf-8') as file:
            self.save_script_to_stream(file)


class WorkflowState(DessiaObject):
    """ State of execution of a workflow. """
    _standalone_in_db = True
    _allowed_methods = ['block_evaluation', 'evaluate_next_block', 'continue_run',
                        'evaluate_maximum_blocks', 'add_block_input_values']
    _non_serializable_attributes = ['activated_items']

    def __init__(self, workflow: Workflow, input_values=None, activated_items=None, values=None,
                 start_time: float = None, end_time: float = None, output_value=None, log: str = '', name: str = ''):
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
        workflow_state = self.__class__(workflow=workflow, input_values=input_values, activated_items=activated_items,
                                        values=values, start_time=self.start_time, end_time=self.end_time,
                                        output_value=deepcopy_value(value=self.output_value, memo=memo),
                                        log=self.log, name=self.name)
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

        for pipe, other_pipe in zip(self.workflow.pipes, other_object.workflow.pipes):
            if self.activated_items[pipe] != other_object.activated_items[other_pipe]:
                # Check pipe progress state
                return False

            if self.activated_items[pipe] and pipe in self.values:
                if self.values[pipe] != other_object.values[other_pipe]:
                    # Check variable values for evaluated ones
                    return False
        return True

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """
        Transform object into a dict.
        """
        if memo is None:
            memo = {}
        id_memo = {}

        if use_pointers:
            workflow_dict = self.workflow.to_dict(path=f'{path}/workflow', memo=memo)
        else:
            workflow_dict = self.workflow.to_dict(use_pointers=False)

        dict_ = self.base_dict()
        dict_.update({'start_time': self.start_time, 'end_time': self.end_time,
                      'log': self.log, "workflow": workflow_dict})
        # Force migrating from dessia_common.workflow
        dict_['object_class'] = 'dessia_common.workflow.core.WorkflowState'

        input_values = {}
        for input_, value in self.input_values.items():
            if use_pointers:
                serialized_v, memo = serialize_with_pointers(value=value, memo=memo,
                                                             path=f"{path}/input_values/{input_}",
                                                             id_memo=id_memo)
            else:
                serialized_v = serialize(value)
            input_values[str(input_)] = serialized_v

        dict_['input_values'] = input_values

        # Output value: priority for reference before values
        if self.output_value is not None:
            if use_pointers:
                serialized_output_value, memo = serialize_with_pointers(self.output_value, memo=memo,
                                                                        path=f'{path}/output_value',
                                                                        id_memo=id_memo)
            else:
                serialized_output_value = serialize(self.output_value)

            dict_.update({'output_value': serialized_output_value,
                          'output_value_type': recursive_type(self.output_value)})

        # Values
        values = {}
        for pipe, value in self.values.items():
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
        dict_['values'] = values

        # In the future comment these below and rely only on activated items
        dict_['evaluated_blocks_indices'] = [i for i, b in enumerate(self.workflow.blocks)
                                             if b in self.activated_items and self.activated_items[b]]

        dict_['evaluated_pipes_indices'] = [i for i, p in enumerate(self.workflow.pipes)
                                            if p in self.activated_items and self.activated_items[p]]

        dict_['evaluated_variables_indices'] = [self.workflow.variable_indices(v) for v in self.workflow.variables
                                                if v in self.activated_items and self.activated_items[v]]
        dict_["_references"] = id_memo
        # Uncomment when refs are handled as dict keys
        # activated_items = {}
        # for key, activated in self.activated_items.items():
        #     s_key, memo = serialize_with_pointers(key, memo=memo, path=f'{path}/activated_items/{key}')
        #     print('s_key', s_key)
        #     activated_items[s_key] = activated
        return dict_

    def state_display(self):
        """
        Compute display.

        TODO This doesn't compute display at all. It probably was the reason of display failure. Copy/Paste problem ?
        """
        memo = {}

        workflow_dict = self.workflow.to_dict(path='#/workflow', memo=memo)

        dict_ = self.base_dict()
        # Force migrating from dessia_common.workflow
        dict_['object_class'] = 'dessia_common.workflow.core.WorkflowState'

        dict_['workflow'] = workflow_dict

        dict_['filled_inputs'] = list(sorted(self.input_values.keys()))

        # Output value: priority for reference before values
        if self.output_value is not None:
            serialized_output_value, memo = serialize_with_pointers(self.output_value, memo=memo, path='#/output_value')
            dict_['output_value'] = serialized_output_value

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
        """ Compute Workflow State from diven dict. Handles pointers. """
        if pointers_memo is None or global_dict is None:
            global_dict, pointers_memo = update_pointers_data(global_dict=global_dict, current_dict=dict_,
                                                              pointers_memo=pointers_memo)

        workflow = Workflow.dict_to_object(dict_=dict_['workflow'], global_dict=global_dict,
                                           pointers_memo=pointers_memo, path=f"{path}/workflow")
        if 'output_value' in dict_:
            value = dict_['output_value']
            output_value = deserialize(value, global_dict=global_dict,
                                       pointers_memo=pointers_memo, path=f'{path}/output_value')
        else:
            output_value = None

        values = {}
        if 'values' in dict_:
            for i, value in dict_['values'].items():
                values[workflow.pipes[int(i)]] = deserialize(value, global_dict=global_dict,
                                                             pointers_memo=pointers_memo, path=f'{path}/values/{i}')

        input_values = {int(i): deserialize(v, global_dict=global_dict, pointers_memo=pointers_memo,
                                            path=f"{path}/input_values/{i}") for i, v in dict_['input_values'].items()}

        activated_items = {b: i in dict_['evaluated_blocks_indices'] for i, b in enumerate(workflow.blocks)}
        activated_items.update({p: i in dict_['evaluated_pipes_indices'] for i, p in enumerate(workflow.pipes)})

        var_indices = []
        for variable_indices in dict_['evaluated_variables_indices']:
            if is_sequence(variable_indices):
                var_indices.append(tuple(variable_indices))  # json serialisation loses tuples
            else:
                var_indices.append(variable_indices)
        activated_items.update({v: workflow.variable_indices(v) in var_indices for v in workflow.variables})

        return cls(workflow=workflow, input_values=input_values, activated_items=activated_items,
                   values=values, start_time=dict_['start_time'], end_time=dict_['end_time'],
                   output_value=output_value, log=dict_['log'], name=dict_['name'])

    def add_input_value(self, input_index: int, value):
        """Add a value for given input."""
        self._activate_input(input_=self.workflow.inputs[input_index], value=value)

    def add_several_input_values(self, indices: List[int], values):
        """
        Add several values for given inputs.
        """
        for index in indices:
            input_ = self.workflow.inputs[index]
            if index not in values:
                if self.activated_items[input_] and index in self.input_values:
                    value = self.input_values[index]
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
        Add inputs values for given block.
        """
        values = {int(k): v for k, v in values.items()}  # serialisation set keys as strings
        indices = self.workflow.block_inputs_global_indices(block_index)
        self.add_several_input_values(indices=indices, values=values)

    def display_settings(self) -> List[DisplaySetting]:
        """
        Compute the displays settings of the objects.
        """
        display_settings = [DisplaySetting('workflow-state', 'workflow_state', 'state_display', None)]

        # Displayable blocks
        display_settings.extend(self.workflow.blocks_display_settings)
        return display_settings

    def _display_from_selector(self, selector: str, **kwargs) -> DisplayObject:
        """
        Generate the display from the selector.
        """
        # TODO THIS IS A TEMPORARY DIRTY HOTFIX OVERWRITE.
        #  WE SHOULD IMPLEMENT A WAY TO GET RID OF REFERENCE PATH WITH URLS
        track = ""
        refpath = kwargs.get("reference_path", "")
        if selector in ["documentation", "workflow"]:
            return self.workflow._display_from_selector(selector)

        if selector == "workflow-state":
            return DessiaObject._display_from_selector(self, selector)

        # Displays for blocks (getting reference path from block_display return)
        display_setting = self._display_settings_from_selector(selector)
        try:
            # Specific hotfix : we propagate reference_path through block_display method
            display_object, refpath = attrmethod_getter(self, display_setting.method)(**display_setting.arguments)
            data = display_object.data
        except:
            data = None
            track = tb.format_exc()

        if display_setting.serialize_data:
            data = serialize(data)
        return DisplayObject(type_=display_setting.type, data=data, reference_path=refpath, traceback=track)

    def block_display(self, block_index: int):
        """
        Compute the display of associated block to use integrate it in the workflow run displays.
        """
        self.activate_inputs()
        block = self.workflow.blocks[block_index]

        selector = self.workflow.block_selectors[block]
        branch = self.workflow.branch_by_display_selector[selector]
        evaluated_blocks = self.evaluate_branch(branch)

        reference_path = ""
        for i, input_ in enumerate(block.inputs):
            incoming_pipe = self.workflow.variable_input_pipe(input_)
            if i == block._displayable_input:
                reference_path = f'values/{self.workflow.pipes.index(incoming_pipe)}'

        if block not in evaluated_blocks:
            msg = f"Could not reach block at index {block_index}." \
                  f"Has the workflow been run far enough to evaluate this block ?"
            raise WorkflowError(msg)
        return evaluated_blocks[block][0], reference_path  # Only one output to an Export Block

    @property
    def progress(self):
        """
        Return the current progress, a float between 0 (nothing evaluated), to 1. (every computational block evaluated).
        """
        evaluated_blocks = [self.activated_items[b] for b in self.workflow.runtime_blocks]
        progress = sum(evaluated_blocks) / len(evaluated_blocks)
        if progress == 1 and self.end_time is None:
            self.end_time = time.time()
        return progress

    def block_evaluation(self, block_index: int, progress_callback=lambda x: None) -> bool:
        """
        Select a block to evaluate.
        """
        block = self.workflow.blocks[block_index]
        self.activate_inputs()
        if block in self._activable_blocks():
            self._evaluate_block(block)
            progress_callback(self.progress)
            return True
        return False

    def evaluate_next_block(self, progress_callback=lambda x: None) -> Optional[Block]:
        """
        Evaluate a block.
        """
        self.activate_inputs()
        blocks = self._activable_blocks()
        if blocks:
            block = blocks[0]
            self._evaluate_block(block)
            progress_callback(self.progress)
            return block
        return None

    def continue_run(self, progress_callback=lambda x: None, export: bool = False):
        """
        Evaluate all possible blocks.
        """
        self.activate_inputs()

        evaluated_blocks = []
        something_activated = True
        while something_activated:
            something_activated = False
            blocks = [b for b in self.workflow.runtime_blocks if b in self._activable_blocks()]
            for block in blocks:
                evaluated_blocks.append(block)
                self._evaluate_block(block)
                if not export:
                    progress_callback(self.progress)
                something_activated = True
        return evaluated_blocks

    def evaluate_branch(self, blocks: List[Block]):
        """
        Evaluate all blocks of a branch, automatically finding the first executable ones.
        """
        self.activate_inputs()

        if not any((b in self._activable_blocks() for b in blocks)):
            raise WorkflowError("Branch cannot be evaluated because no block has all its inputs activated")

        evaluated_blocks = {}
        i = 0
        while len(evaluated_blocks) != len(blocks) and i <= len(blocks):
            next_blocks = [b for b in blocks if b in self._activable_blocks() and b not in evaluated_blocks]
            for block in next_blocks:
                output_values = self._evaluate_block(block)
                evaluated_blocks[block] = output_values
            i += 1
        return evaluated_blocks

    def _activate_pipe(self, pipe: Pipe, value):
        """
        Set the pipe value and activate its downstream variable.
        """
        self.values[pipe] = value
        self.activated_items[pipe] = True
        self._activate_variable(variable=pipe.output_variable, value=value)

    def _activate_block(self, block: Block, output_values):
        """
        Activate all block outputs.
        """
        # Unpacking result of evaluation
        output_items = zip(block.outputs, output_values)
        for output, output_value in output_items:
            self._activate_variable(variable=output, value=output_value)
        self.activated_items[block] = True

    def _activate_variable(self, variable: Variable, value):
        """
        Activate the given variable with its value and propagate activation to its outgoing pipe.
        """
        outgoing_pipes = self.workflow.variable_output_pipes(variable)
        if self.workflow.output == variable:
            self.output_value = value
        for outgoing_pipe in outgoing_pipes:
            self._activate_pipe(pipe=outgoing_pipe, value=value)
        self.activated_items[variable] = True

    def _activate_input(self, input_: TypedVariable, value):  # Inputs must always be Typed
        """
        Typecheck, activate the variable and propagate the value to its pipe.
        """
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
        Return a list of all activable blocks, ie blocks that have all inputs ready for evaluation.
        """
        return [b for b in self.workflow.blocks if self._block_activable_by_inputs(b)
                and (not self.activated_items[b] or b not in self.workflow.runtime_blocks)]

    def _block_activable_by_inputs(self, block: Block):
        """
        Return wether a block has all its inputs active and can be activated.
        """
        for function_input in block.inputs:
            if not self.activated_items[function_input]:
                return False
        return True

    def _evaluate_block(self, block, progress_callback=lambda x: x, verbose=False):
        """
        Evaluate given block.
        """
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

        output_values = block.evaluate(local_values)
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
            elif variable in self.workflow.imposed_variable_values:
                self._activate_input(input_=variable, value=self.workflow.imposed_variable_values[variable])
            elif hasattr(variable, 'default_value'):
                self._activate_input(input_=variable, value=variable.default_value)
            elif check_all_inputs:
                msg = f"Value {variable.name} of index {index} in inputs has no value"
                if isinstance(variable, TypedVariable):
                    msg += f": should be instance of {variable.type_}"
                raise ValueError(msg)

    def to_workflow_run(self, name: str = ""):
        """ Return a WorkflowRun if state is complete. """
        if self.progress == 1:
            values = {p: self.values[p] for p in self.workflow.pipes if p in self.values}
            return WorkflowRun(workflow=self.workflow, input_values=self.input_values, output_value=self.output_value,
                               values=values, activated_items=self.activated_items, start_time=self.start_time,
                               end_time=self.end_time, log=self.log, name=name)
        raise ValueError('Workflow not completed')

    def _export_formats(self):
        """ Read block to compute available export formats. """
        export_formats = DessiaObject._export_formats(self)

        # Exportable Blocks
        export_formats.extend(self.workflow.blocks_export_formats)
        return export_formats

    def export_format_from_selector(self, selector: str):
        """ Get WorflowState format from given selector. """
        for export_format in self.workflow.blocks_export_formats:
            if export_format["selector"] == selector:
                return export_format
        raise ValueError(f"No block defines an export with the selector '{selector}'")

    def export(self, stream: Union[BinaryFile, StringFile], block_index: int):
        """ Perform export. """
        block = self.workflow.blocks[block_index]
        selector = self.workflow.block_selectors[block]
        branch = self.workflow.branch_by_export_format[selector]
        evaluated_blocks = self.evaluate_branch(branch)
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


class WorkflowRun(WorkflowState):
    """ Completed state of a workflow. """
    _standalone_in_db = True
    _allowed_methods = ['run_again']
    _eq_is_data_eq = True
    _jsonschema = {
        "definitions": {}, "$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
        "standalone_in_db": True, "title": "WorkflowRun Base Schema", "required": [],
        "python_typing": 'dessia_common.workflow.WorkflowRun', "classes": ["dessia_common.workflow.core.WorkflowRun"],
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
        filtered_values = {p: v for p, v in values.items() if is_serializable(v) and p.memorize}
        filtered_input_values = {k: v for k, v in input_values.items() if is_serializable(v)}
        WorkflowState.__init__(self, workflow=workflow, input_values=filtered_input_values,
                               activated_items=activated_items, values=filtered_values, start_time=start_time,
                               output_value=output_value, log=log, name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True, id_memo=None):
        """ Add variable values to super WorkflowState dict. """
        if memo is None:
            memo = {}  # To make sure we have the good ref for next steps
        dict_ = WorkflowState.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)

        # To force migrating from dessia_common.workflow
        dict_['object_class'] = 'dessia_common.workflow.core.WorkflowRun'
        return dict_

    def _get_from_path(self, path: str):
        """
        Extract subobject at given path. Tries the generic function, then applies specific cases if it fails.

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
        """ Compute run method's args from serialized ones. """
        if method in self._allowed_methods:
            return self.workflow.dict_to_arguments(dict_=dict_, method='run')
        raise NotImplementedError(f"Method {method} not in WorkflowRun allowed methods")

    def display_settings(self) -> List[DisplaySetting]:
        """
        Compute WorkflowRun display settings by getting WorkflowState ones and instering Workflow display settings.
        """
        workflow_settings = self.workflow.display_settings()
        display_settings = WorkflowState.display_settings(self)
        # TODO : Temporary removing workflow state. We could activate it again when display tree is available
        display_settings.pop(0)
        return workflow_settings + display_settings

    def method_dict(self, method_name: str = None, method_jsonschema: Any = None):
        """ Get run again default dict. """
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
        """ Execute workflow again with given inputs. """
        return self.workflow.run(input_values=input_values, verbose=False,
                                 progress_callback=progress_callback, name=name)

    @property
    def _method_jsonschemas(self):
        # TODO This is outdated now that WorkflowRun inherits from WorkflowState and has already broke once.
        #  We should outsource the "run" jsonschema computation from workflow in order to mutualize it with run_again,
        #  and have WorkflowRun have its inheritances from WorkflowState _method_jsonschema method
        workflow_jsonschemas = self.workflow._method_jsonschemas
        jsonschemas = {"run_again": workflow_jsonschemas.pop('run')}
        jsonschemas['run_again']['classes'] = ["dessia_common.workflow.WorkflowRun"]
        return jsonschemas


def initialize_workflow(dict_, global_dict, pointers_memo) -> Workflow:
    """ Generate an unfinished workflow in order to get access to instance method before full deserialization. """
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
    return Workflow(blocks=blocks, pipes=pipes, output=output,
                    detached_variables=[v for v, is_connected in connected_nbvs.items() if not is_connected])


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

        pipes.append(Pipe(variable1, variable2))
    return pipes


def value_type_check(value, type_):
    """ Check if the value as the specified type. """
    try:  # TODO: Subscripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass
    return True
