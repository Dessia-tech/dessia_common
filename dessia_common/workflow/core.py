#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gathers all workflow relative features
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
from dessia_common.templates import workflow_template
from dessia_common import DessiaObject, is_sequence, JSONSCHEMA_HEADER, jsonschema_from_annotation,\
    deserialize_argument, set_default_value, prettyname, serialize_dict, DisplaySetting

from dessia_common.utils.serialization import dict_to_object, deserialize, serialize_with_pointers, serialize,\
                                              dereference_jsonpointers
from dessia_common.utils.types import serialize_typing, deserialize_typing, recursive_type, typematch
from dessia_common.utils.copy import deepcopy_value
from dessia_common.utils.docstrings import FAILED_ATTRIBUTE_PARSING, EMPTY_PARSED_ATTRIBUTE
from dessia_common.utils.diff import choose_hash
from dessia_common.typings import JsonSerializable, MethodType
from dessia_common.displays import DisplayObject
from dessia_common.breakdown import attrmethod_getter


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
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'TypedVariable':
        type_ = deserialize_typing(dict_['type_'])
        memorize = dict_['memorize']
        return cls(type_=type_, memorize=memorize, name=dict_['name'])

    def copy(self, deep: bool = False, memo=None):
        return TypedVariable(type_=self.type_, memorize=self.memorize, name=self.name)

    def to_script(self, variable_index: int):
        script = f"variable_{variable_index} = TypedVariable(type={serialize_typing(self.type_)}, "
        script += f"memorize={self.memorize}, name='{self.name}')\n"
        return script


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
                       pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'TypedVariableWithDefaultValue':
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
        if memo is None:
            memo = {}
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

    def __init__(self, inputs: List[Variable], outputs: List[Variable],
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

    def _docstring(self):
        """
        Base function for submodel docstring computing
        """
        block_docstring = {i: EMPTY_PARSED_ATTRIBUTE for i in self.inputs}
        return block_docstring


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
        "classes": ["dessia_common.workflow.core.Pipe"], "required": ["input_variable", "output_variable"],
        "properties": {
            "input_variable": {
                "type": "object", "editable": True, "order": 0,
                "python_typing": "List[dessia_common.workflow.Variable]",
                "classes": ["dessia_common.workflow.Variable", "dessia_common.workflow.TypedVariable",
                            "dessia_common.workflow.VariableWithDefaultValue",
                            "dessia_common.workflow.TypedVariableWithDefaultValue"]
            },
            "output_variable": {
                "type": "object", "editable": True, "order": 1,
                "python_typing": "List[dessia_common.workflow.Variable]",
                "classes": ["dessia_common.workflow.Variable", "dessia_common.workflow.TypedVariable",
                            "dessia_common.workflow.VariableWithDefaultValue",
                            "dessia_common.workflow.TypedVariableWithDefaultValue"],
            },
            "name": {'type': 'string', 'title': 'Name', 'editable': True,
                     'order': 2, 'default_value': '', 'python_typing': 'builtins.str'}
        }
    }
    _eq_is_data_eq = False

    def __init__(self, input_variable: Variable, output_variable: Variable, name: str = ''):
        self.input_variable = input_variable
        self.output_variable = output_variable
        DessiaObject.__init__(self, name=name)

    def to_dict(self, use_pointers=True, memo=None, path: str = '#'):
        """
        transform the pipe into a dict
        """
        return {'input_variable': self.input_variable, 'output_variable': self.output_variable}

    @staticmethod
    def to_script(pipe_index: int, input_name: str, output_name: str):
        """
        Transform the pipe into a little chunk of code
        """
        script = f"pipe_{pipe_index} = dcw.Pipe({input_name}, {output_name})"
        return script


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
            except ValueError as err:
                raise ValueError(f"Cannot serialize block {block} ({block.name})") from err

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

        if output is not None:
            output.memorize = True

        Block.__init__(self, input_variables, [output], name=name)
        self.output = self.outputs[0]

    def _data_hash(self):
        output_hash = hash(self.variable_indices(self.outputs[0]))

        base_hash = len(self.blocks) + 11 * len(self.pipes) + output_hash
        block_hash = int(sum(b.equivalent_hash() for b in self.blocks) % 10e5)
        return (base_hash + block_hash) % 1000000000

    def _data_eq(self, other_object):  # TODO: implement imposed_variable_values in equality
        if hash(self) != hash(other_object) or not Block.equivalent(self, other_object):
            return False
        # TODO: temp , reuse graph!!!!
        for block1, block2 in zip(self.blocks, other_object.blocks):
            if not block1.equivalent(block2):
                return False

        if len(self.imposed_variable_values) != len(other_object.imposed_variable_values):
            return False
        for imposed_key1, imposed_key2 in zip(self.imposed_variable_values.keys(),
                                              other_object.imposed_variable_values.keys()):
            if hash(imposed_key1) != hash(imposed_key2):
                return False
            imposed_value1 = self.imposed_variable_values[imposed_key1]
            imposed_value2 = other_object.imposed_variable_values[imposed_key2]
            if hash(imposed_value1) != hash(imposed_value2):
                return False

        return True

    def __deepcopy__(self, memo=None):
        """
        Returns the deep copy
        """
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

        pipes = [self.copy_pipe(pipe=p, copied_workflow=copied_workflow) for p in self.pipes]

        imposed_variable_values = {}
        for variable, value in self.imposed_variable_values.items():
            new_variable = copied_workflow.variable_from_index(self.variable_indices(variable))
            imposed_variable_values[new_variable] = value

        copied_workflow = Workflow(blocks=blocks, pipes=pipes, output=output,
                                   imposed_variable_values=imposed_variable_values, name=self.name)
        return copied_workflow

    def copy_pipe(self, pipe: Pipe, copied_workflow: 'Workflow') -> Pipe:
        """
        Copy a pipe to another workflow
        """
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

    @staticmethod
    def display_settings() -> List[DisplaySetting]:
        """
        Computes the displays of the objects
        """
        display_settings = [DisplaySetting('documentation', 'markdown', 'to_markdown', None),
                            DisplaySetting('workflow', 'workflow', 'to_dict', None)]
        return display_settings

    def to_markdown(self):
        """
        Sets workflow documentation as markdown
        """
        return self.documentation
        # return DisplayObject(type_="markdown", data=self.documentation)

    def _docstring(self):
        """
        Computes documentation of all blocks
        """
        docstrings = [b._docstring() for b in self.blocks]
        return docstrings

    @property
    def _method_jsonschemas(self):
        """
        Compute the run jsonschema (had to be overloaded)
        """
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
            if input_ not in self.imposed_variable_values:  # Removes from Optional in edits
                properties_dict[str(i)] = current_dict[str(i)]
        properties_dict[str(len(self.inputs) + 1)] = {'type': 'string', 'title': 'WorkflowRun Name', 'editable': True,
                                                      'order': 0, "description": "Name for the resulting WorkflowRun",
                                                      'default_value': '', 'python_typing': 'builtins.str'}
        jsonschemas['run'].update({'required': required_inputs, 'method': True,
                                   'python_typing': serialize_typing(MethodType)})
        jsonschemas['start_run'] = deepcopy(jsonschemas['run'])
        jsonschemas['start_run']['required'] = []
        return jsonschemas

    def _export_formats(self):
        """
        Reads block to compute available export formats
        """
        export_formats = DessiaObject._export_formats(self)
        export_formats.append({'extension': 'py', 'method_name': 'save_script_to_stream',
                               'text': True, 'args': {}})
        return export_formats

    def to_dict(self, use_pointers=True, memo=None, path='#'):
        """
        Compute a dict from the object content
        """
        if memo is None:
            memo = {}

        self.refresh_blocks_positions()
        dict_ = Block.to_dict(self)
        dict_['object_class'] = 'dessia_common.workflow.core.Workflow'  # TO force migrating from dessia_common.workflow
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
            imposed_variable_values[str(var_index)] = ser_value

        dict_.update({'description': self.description, 'documentation': self.documentation,
                      'imposed_variable_values': imposed_variable_values})
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False,
                       global_dict=None, pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'Workflow':
        """
        Recompute the object from a dict
        """
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

        if dict_['output'] is not None:
            output = blocks[dict_['output'][0]].outputs[dict_['output'][2]]
        else:
            output = None
        temp_workflow = cls(blocks=blocks, pipes=pipes, output=output)

        if 'imposed_variable_values' in dict_ and 'imposed_variables' in dict_:
            # Legacy support of double list
            imposed_variable_values = {}
            iterator = zip(dict_['imposed_variables'], dict_['imposed_variable_values'])
            for variable_index, serialized_value in iterator:
                value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                variable = temp_workflow.variable_from_index(variable_index)

                imposed_variable_values[variable] = value
        else:
            imposed_variable_values = {}
            if 'imposed_variable_indices' in dict_:
                for variable_index in dict_['imposed_variable_indices']:
                    variable = temp_workflow.variable_from_index(variable_index)
                    imposed_variable_values[variable] = variable.default_value
            if 'imposed_variable_values' in dict_:
                # New format with a dict
                for variable_index_str, serialized_value in dict_['imposed_variable_values'].items():
                    variable_index = ast.literal_eval(variable_index_str)
                    value = deserialize(serialized_value, global_dict=global_dict, pointers_memo=pointers_memo)
                    variable = temp_workflow.variable_from_index(variable_index)
                    imposed_variable_values[variable] = value

            if 'imposed_variable_indices' not in dict_ and 'imposed_variable_values' not in dict_:
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
        """
        Process a json of arguments and deserialize them
        """
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
            if name_index in dict_:
                name = dict_[name_index]
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
        """
        Cached property for graph
        """
        if not self._utd_graph:
            self._cached_graph = self._graph()
            self._utd_graph = True
        return self._cached_graph

    graph = property(_get_graph)

    def _graph(self):
        """
        Compute the networkx graph of the workflow
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
            for candidate in block_upstreams:
                if candidate not in runtime_blocks:
                    runtime_blocks.append(candidate)
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

    def get_upstream_nbv(self, variable: Variable) -> Variable:
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

    def upstream_variable(self, variable: Variable) -> Optional[Variable]:
        """
        Returns upstream variable if given variable is connected to a pipe as a pipe output

        :param variable: Variable to search an upstream for
        """
        incoming_pipes = [p for p in self.pipes if p.output_variable == variable]
        if incoming_pipes:  # Inputs can only be connected to one pipe
            incoming_pipe = incoming_pipes[0]
            return incoming_pipe.input_variable
        return None

    def variable_indices(self, variable: Variable) -> Optional[Union[Tuple[int, int, int], int]]:
        """
        Returns global adress of given variable as a tuple or an int

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

    def block_from_variable(self, variable) -> Block:
        """
        Returns block of which given variable is attached to
        """
        iblock, _, _ = self.variable_indices(variable)
        return self.blocks[iblock]

    def output_disconnected_elements(self):
        """
        Return blocks and variables that are not attached to the output
        """
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
        Deprecated, will be remove in version 0.8.0
        """
        warnings.warn("index method is deprecated, use input_index instead", DeprecationWarning)
        return self.input_index(variable)

    def input_index(self, variable: Variable) -> Optional[int]:
        """
        If variable is a workflow input, returns its index
        """
        upstream_variable = self.get_upstream_nbv(variable)
        if upstream_variable in self.inputs:
            return self.inputs.index(upstream_variable)
        return None

    def variable_index(self, variable: Variable) -> int:
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

    def match_variables(self, serialize_output: bool = False):
        """
        Runs a check for every variable to find its matchable counterparts which means :
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
        """
        Recomputes block positions
        """
        coordinates = self.layout()
        for block in self.blocks:
            # TODO merge these two loops
            block.position = coordinates[block]
        for nonblock in self.nonblock_variables:
            nonblock.position = coordinates[nonblock]

    def plot_graph(self):
        """
        Plot graph by means of networking and matplotlib
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
        Full run of a workflow. Yields a WorkflowRun
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

        state.output_value = state.values[self.outputs[0]]

        name_index = str(len(self.inputs) + 1)
        if name is None and name_index in input_values:
            name = input_values[name_index]
        if not name:
            timestamp = start_timestamp.strftime("%m-%d (%H:%M)")
            name = f"{self.name} @ [{timestamp}]"
        return state.to_workflow_run(name=name)

    def start_run(self, input_values=None, name: str = None):
        """
        Partial run of a workflow. Yields a WorkflowState
        """
        return WorkflowState(self, input_values=input_values, name=name)

    def jointjs_data(self):
        """
        Computes the data needed for jointjs ploting
        """
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
        """
        Tell if the workflow is valid:
            * check type compatibility of pipes inputs/outputs
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

    def to_script(self) -> str:
        """
        Computes a script representing the workflow.
        """
        variable_index = 0
        classes = []

        script_blocks = ''
        for ib, block in enumerate(self.blocks):
            block_script, classes_block = block.to_script()
            classes.extend(classes_block)
            script_blocks += f'block_{ib} = {block_script}\n'

        script = ('import dessia_common.workflow as dcw\n'
                  + 'import dessia_common.workflow.blocks as dcw_blocks\n'
                  + 'import dessia_common.typings as dct\n\n')

        modules = {'.'.join(c.split('.')[:-1]) for c in classes}
        for module in modules:
            script += f'import {module}\n'
        script += '\n'
        script += script_blocks
        script += 'blocks = [{}]\n'.format(', '.join(['block_' + str(i) for i in range(len(self.blocks))]))

        for ip, pipe in enumerate(self.pipes):
            input_index = self.variable_indices(pipe.input_variable)
            if isinstance(input_index, int):
                script += pipe.input_variable.to_script(variable_index=variable_index) + '\n'
                input_name = f'variable_{variable_index}'
                variable_index += 1
            else:
                input_name = f"block_{input_index[0]}.outputs[{input_index[2]}]" + '\n'

            output_index = self.variable_indices(pipe.output_variable)
            if isinstance(output_index, int):
                script += pipe.output_variable.to_script(variable_index=variable_index) + '\n'
                output_name = f'variable_{variable_index}'
                variable_index += 1
            else:
                output_name = f"block_{output_index[0]}.inputs[{output_index[2]}]" + '\n'
            script += pipe.to_script(pipe_index=ip, input_name=input_name, output_name=output_name) + '\n'

        script += f"pipes = [{', '.join(['pipe_' + str(i) for i in range(len(self.pipes))])}]\n"

        workflow_output_index = self.variable_indices(self.output)
        if workflow_output_index is None:
            raise ValueError("A workflow output must be set")
        output_name = f"block_{workflow_output_index[0]}.outputs[{workflow_output_index[2]}]"
        script += f"workflow = dcw.Workflow(blocks, pipes, output={output_name},name='{self.name}')\n"
        return script

    def save_script_to_stream(self, stream: io.StringIO):
        """
        Save the workflow to a python script to a stream
        """
        string = self.to_script()
        stream.seek(0)
        stream.write(string)

    def save_script_to_file(self, filename: str):
        """
        Save the workflow to a python script to a file on the disk
        """
        if not filename.endswith('.py'):
            filename += '.py'

        with open(filename, 'w', encoding='utf-8') as file:
            self.save_script_to_stream(file)


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
        progress = int(100 * self.progress)
        workflow = hash(self.workflow)
        output = choose_hash(self.output_value)
        input_values = sum(i * choose_hash(v) for (i, v) in self.input_values.items())
        values = sum(len(k.name) * choose_hash(v) for (k, v) in self.values.items())
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
        # Force migrating from dessia_common.workflow
        dict_['object_class'] = 'dessia_common.workflow.core.WorkflowState'

        dict_['workflow'] = workflow_dict

        input_values = {}
        for input_, value in self.input_values.items():
            if use_pointers:
                serialized_v, memo = serialize_with_pointers(value=value, memo=memo,
                                                             path=f"{path}/input_values/{input_}")
            else:
                serialized_v = serialize(value)
            input_values[str(input_)] = serialized_v

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
                values[str(variable_index)] = serialized_value
        else:
            values = {str(self.workflow.variable_index(i)): serialize(v) for i, v in self.values.items()}

        dict_['values'] = values

        # In the future comment these below and rely only on activated items
        dict_['evaluated_blocks_indices'] = [i for i, b in enumerate(self.workflow.blocks)
                                             if b in self.activated_items and self.activated_items[b]]

        dict_['evaluated_pipes_indices'] = [i for i, p in enumerate(self.workflow.pipes)
                                            if p in self.activated_items and self.activated_items[p]]

        dict_['evaluated_variables_indices'] = [self.workflow.variable_indices(v) for v in self.workflow.variables
                                                if v in self.activated_items and self.activated_items[v]]

        # Uncomment when refs are handled as dict keys
        # activated_items = {}
        # for key, activated in self.activated_items.items():
        #     s_key, memo = serialize_with_pointers(key, memo=memo, path=f'{path}/activated_items/{key}')
        #     print('s_key', s_key)
        #     activated_items[s_key] = activated

        dict_.update({'start_time': self.start_time, 'end_time': self.end_time, 'log': self.log})
        return dict_

    def state_display(self):
        """
        Compute display
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
            serialized_output_value, memo = serialize_with_pointers(self.output_value, memo=memo,
                                                                    path='#/output_value')
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

        values = {}
        if 'values' in dict_:
            for i, value in dict_['values'].items():
                values[workflow.variables[int(i)]] = deserialize(value, global_dict=dict_,
                                                                 pointers_memo=pointers_memo,
                                                                 path=f'{path}/values/{i}')
        # elif 'variable_values' in dict_:
        #     for i, value in dict_['variable_values'].items():
        #         values[workflow.variables[int(i)]] = deserialize(value, global_dict=dict_,
        #                                                          pointers_memo=pointers_memo,
        #                                                          path=f'{path}/variable_values/{i}')

        input_values = {int(i): deserialize(v, global_dict=dict_, pointers_memo=pointers_memo,
                                            path=f"{path}/input_values/{i}")
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

    @staticmethod
    def display_settings() -> List[DisplaySetting]:
        """
        Computes the displays of the objects
        """
        return [DisplaySetting('workflow-state', 'workflow_state', 'state_display', None)]

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
        while something_activated:  # and (self.progress < 1 or export)
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
        for pipe in self.workflow.pipes:
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
            elif variable in self.workflow.imposed_variable_values:
                self.values[variable] = self.workflow.imposed_variable_values[variable]
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
        export_formats = DessiaObject._export_formats(self)
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
        self.variable_values = {workflow.variable_indices(k): v for k, v in values.items() if k.memorize}
        WorkflowState.__init__(self, workflow=workflow, input_values=input_values, activated_items=activated_items,
                               values=values, start_time=start_time, output_value=output_value, log=log, name=name)

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
        """
        Adds variable values to super WorkflowState dict
        """
        if memo is None:
            memo = {}  # To make sure we have the good ref for next steps
        dict_ = WorkflowState.to_dict(self, use_pointers=use_pointers, memo=memo, path=path)

        # To force migrating from dessia_common.workflow
        dict_['object_class'] = 'dessia_common.workflow.core.WorkflowRun'

        # TODO REMOVING THIS TEMPORARLY TO PREVENT DISPLAYS TO BE LOST WITH POINTERS
        # VARIABLE_VALUES ARE NOT SET BACK IN DICT_TO_OBJECT
        # if use_pointers:
        #     variable_values = {}
        #     for key, value in self.variable_values.items():
        #         variable_values[str(key)], memo = serialize_with_pointers(value, memo=memo,
        #                                                                   path=f'{path}/variable_values/{key}')
        #     dict_["variable_values"] = variable_values
        # else:
        dict_["variable_values"] = {str(k): serialize(v) for k, v in self.variable_values.items()}
        return dict_

    def display_settings(self) -> List[DisplaySetting]:
        """
        Computes the displays settings of the objects
        """
        display_settings = self.workflow.display_settings()

        # display_settings.append(DisplaySetting('workflow-state', 'workflow_state', 'state_display', None))

        # Find & order displayable blocks
        d_blocks = [b for b in self.workflow.blocks if hasattr(b, 'display_') and hasattr(b, "_display_settings")]
        # Change last line to isinstance ? Not possible because of circular imports ?
        sorted_d_blocks = sorted(d_blocks, key=lambda b: b.order)
        for block in sorted_d_blocks:
            block_index = self.workflow.blocks.index(block)
            local_values = {}
            for i, input_ in enumerate(block.inputs):
                input_adress = self.workflow.variable_indices(input_)
                local_values[input_] = self.variable_values[input_adress]
            settings = block._display_settings(block_index, local_values)  # Code intel is not working properly here
            if settings is not None:
                display_settings.extend(settings)

        if isinstance(self.output_value, DessiaObject):
            output_display_settings = [ds.compose(attribute='output_value', serialize_data=True)
                                       for ds in self.output_value.display_settings()]
            display_settings.extend(output_display_settings)

        return display_settings

    def _display_from_selector(self, selector: str, **kwargs) -> DisplayObject:
        """
        Generate the display from the selector
        """
        # TODO THIS IS A TEMPORARY DIRTY HOTFIX OVERWRITE.
        #  WE SHOULD IMPLEMENT A WAY TO GET RID OF REFERENCE PATH WITH URLS
        track = ""
        if "reference_path" in kwargs:
            refpath = kwargs["reference_path"]
        else:
            refpath = ""
        if selector in ["documentation", "workflow"]:
            return self.workflow._display_from_selector(selector)

        if selector == "workflow-state":
            return DessiaObject._display_from_selector(self, selector)

        # Displays for blocks (getting reference path from block_display return)
        display_setting = self._display_settings_from_selector(selector)
        try:
            if display_setting.method == "block_display":
                # Specific hotfix : we propagate reference_path through block_display method
                display_object, refpath = attrmethod_getter(self, display_setting.method)(**display_setting.arguments)
                data = display_object["data"]
            else:
                # But not when calling result objects display methods.
                # We end up here when evaluating output value display
                data = attrmethod_getter(self, display_setting.method)(**display_setting.arguments)
        except:
            data = None
            track = tb.format_exc()

        if display_setting.serialize_data:
            data = serialize(data)
        return DisplayObject(type_=display_setting.type, data=data, reference_path=refpath, traceback=track)

    def block_display(self, block_index: int, display_index: int):
        """
        Computes the display of associated block to use integrate it in the workflow run displays
        """
        self._activate_activable_pipes()
        self.activate_inputs()
        block = self.workflow.blocks[block_index]
        if block in self._activable_blocks():
            self._evaluate_block(block)
        reference_path = ""
        local_values = {}
        for i, input_ in enumerate(block.inputs):
            input_adress = self.workflow.variable_indices(input_)
            local_values[input_] = self.variable_values[input_adress]
            if i == block._displayable_input:
                reference_path = f'variable_values/{input_adress}'
        display_ = block.display_(local_values=local_values, reference_path=reference_path)
        return display_[display_index], reference_path

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
        # TODO This is outdated now that WorkflowRun inherits from WorkflowState and has already broke once.
        #  We should outsource the "run" jsonschema computation from workflow in order to mutualize it with run_again,
        #  and have WorkflowRun have its inheritances from WorkflowState _method_jsonschema method
        workflow_jsonschemas = self.workflow._method_jsonschemas
        jsonschemas = {"run_again": workflow_jsonschemas.pop('run')}
        jsonschemas['run_again']['classes'] = ["dessia_common.workflow.WorkflowRun"]
        return jsonschemas


def value_type_check(value, type_):
    try:  # TODO: Subscripted generics cannot be used...
        if not isinstance(value, type_):
            return False
    except TypeError:
        pass
    return True
