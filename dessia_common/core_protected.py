#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import inspect
from copy import deepcopy

from typing import TypeVar
from mypy_extensions import _TypedDictMeta
import dessia_common.core

class DessiaObject:
    """

    """

    @classmethod
    def base_jsonschema(cls):
        jsonschema = deepcopy(JSONSCHEMA_HEADER)
        jsonschema['properties']['name'] = {
            'type': 'string',
            "title" : "Object Name",
            "description" : "Object name",
            "editable" : True,
            "default_value" : "Object Name"
            }
        return jsonschema


    @classmethod
    def jsonschema(cls):
        if hasattr(cls, '_jsonschema'):
            _jsonschema = cls._jsonschema
            return _jsonschema

        # Get __init__ method and its annotations
        init = cls.__init__
        if cls._init_variables is None:
            annotations = init.__annotations__
        else:
            annotations = cls._init_variables

        # Get ordered variables
        if cls._ordered_attributes:
            ordered_attributes = cls._ordered_attributes
        else:
            ordered_attributes = list(annotations.keys())

        unordered_count = 0

        # Initialize jsonschema
        _jsonschema = deepcopy(JSONSCHEMA_HEADER)

        required_arguments, default_arguments = inspect_arguments(init, merge=False)
        _jsonschema['required'] = required_arguments

        # Set jsonschema
        for annotation in annotations.items():
            name = annotation[0]
            if name in ordered_attributes:
                order = ordered_attributes.index(name)
            else:
                order = len(ordered_attributes) + unordered_count
                unordered_count += 1
            if name in cls._titled_attributes:
                title = cls._titled_attributes[name]
            else:
                title = None

            if name != 'return':
                
                annotation = (annotation[0], dessia_common.core.type_from_annotation(annotation[1], cls))
                jsonschema_element = jsonschema_from_annotation(annotation=annotation,
                                                                jsonschema_element={},
                                                                order=order,
                                                                editable=name not in cls._non_editable_attributes,
                                                                title=title)
                _jsonschema['properties'].update(jsonschema_element)
                if name in default_arguments.keys():
                    default = set_default_value(_jsonschema['properties'],
                                                name,
                                                default_arguments[name])
                    _jsonschema['properties'].update(default)
                _jsonschema['classes'] = [cls.__module__ + '.' + cls.__name__]
                _jsonschema['whitelist_attributes'] = cls._whitelist_attributes
        return _jsonschema

    @property
    def _method_jsonschemas(self):
        """
        Generates dynamic jsonschemas for methods of class
        """
        jsonschemas = {}
        class_ = self.__class__

        # !!! Backward compatibility. Will need to be changed
        if hasattr(class_, '_dessia_methods'):
            allowed_methods = class_._dessia_methods
        else:
            allowed_methods = class_._allowed_methods

        valid_method_names = [m for m in dir(class_)\
                              if not m.startswith('_')
                              and m in allowed_methods]

        for method_name in valid_method_names:
            method = getattr(class_, method_name)

            if not isinstance(method, property):
                required_arguments, default_arguments = inspect_arguments(method, merge=False)

                if method.__annotations__:
                    jsonschemas[method_name] = deepcopy(JSONSCHEMA_HEADER)
                    jsonschemas[method_name]['required'] = []
                    for i, annotation in enumerate(method.__annotations__.items()): # !!! Not actually ordered
                        argname = annotation[0]
                        if argname not in _FORBIDDEN_ARGNAMES:
                            if argname in required_arguments:
                                jsonschemas[method_name]['required'].append(str(i))
                            jsonschema_element = jsonschema_from_annotation(annotation, {}, i)[argname]

                            jsonschemas[method_name]['properties'][str(i)] = jsonschema_element
                            if argname in default_arguments.keys():
                                default = set_default_value(jsonschemas[method_name]['properties'],
                                                            str(i),
                                                            default_arguments[argname])
                                jsonschemas[method_name]['properties'].update(default)
        return jsonschemas

    def dict_to_arguments(self, dict_, method):
        method_object = getattr(self, method)
        args_specs = inspect.getfullargspec(method_object)
        allowed_args = args_specs.args[1:]
        # self_index = allowed_args.index('self')
        # allowed_args.pop(self_index)

        arguments = {}
        for i, arg in enumerate(allowed_args):
            if str(i) in dict_:
                value = dict_[str(i)]
                try:
                    deserialized_value = deserialize_argument(args_specs.annotations[arg], value)
                except TypeError:
                    raise TypeError('Error in deserialisation of value: {} of expected type {}'.format(value, args_specs.annotations[arg]))
                arguments[arg] = deserialized_value
        return arguments


def jsonschema_from_annotation(annotation, jsonschema_element,
                               order, editable=None, title=None):
    key, value = annotation
    if isinstance(value, str):
        raise ValueError
        
    if title is None:
        title = prettyname(key)
    if editable is None:
        editable = key not in ['return']

    if value in TYPING_EQUIVALENCES.keys():
        # Python Built-in type
        jsonschema_element[key] = {'type': TYPING_EQUIVALENCES[value],
                                   'title': title,
                                   'editable': editable,
                                   'order': order}
    # elif isinstance(value, TypeVar):
    #     # !!! Obsolete, this will probably need to be removed. New way : Union
    #     # Several  classes are possible
    #     classnames = [c.__module__+'.'+c.__name__ for c in value.__constraints__]
    #     jsonschema_element[key] = {'type': 'object',
    #                                'classes': classnames,
    #                                'title': title,
    #                                'editable': editable,
    #                                'order': order}
    elif hasattr(value, '_name') and value._name == 'Union':
        # Types union
        classnames = [c.__module__ + '.' + c.__name__ for c in value.__constraints__]
        jsonschema_element[key] = {'type': 'object',
                                   'classes': classnames,
                                   'title': title,
                                   'editable': editable,
                                   'order': order}
    elif hasattr(value, '_name') and value._name in ['List', 'Sequence', 'Iterable']:
        # Homogenous lists
        jsonschema_element[key] = jsonschema_sequence_recursion(value=value,
                                                                order=order,
                                                                title=title,
                                                                editable=editable)
    elif hasattr(value, '_name') and value._name == 'Tuple':
        # Heterogenous lists
        items = []
        for type_ in value.__args__:
            items.append({'type': TYPING_EQUIVALENCES[type_]})
        jsonschema_element[key] = {'additionalItems': False,
                                   'type': 'array'}
        jsonschema_element[key]['items'] = items
    elif hasattr(value, '_name') and value._name == 'Dict':
        # Dynamially created dict structure
        key_type, value_type = value.__args__
        if key_type != str:
            raise NotImplementedError('Non strings keys not supported')  # !!! Should we support other types ? Numeric ?
        jsonschema_element[key] = {'type': 'object',
                                   'order': order,
                                   'editable': editable,
                                   'title': title,
                                   'patternProperties': {'.*': {'type': TYPING_EQUIVALENCES[value_type]}}}
    elif isinstance(value, _TypedDictMeta):
        # Static dict structure
        print(value.__dict__)
        # !!! This will actually some investigations on wether we use dataclasses or not, as TypedDict can't be check with insinstance
        raise NotImplementedError
    else:
        # Custom classes
        if hasattr(value, '_standalone_in_db'):  # and value._standalone_in_db:
            # Dessia custom classes
            classname = value.__module__ + '.' + value.__name__
            jsonschema_element[key] = {'type': 'object',
                                       'title': title,
                                       'order': order,
                                       'editable': editable,
                                       'classes': [classname]}
        else:
            # Dataclasses
            jsonschema_element[key] = jsonschema_from_dataclass(value)
            jsonschema_element[key].update({'title': title,
                                            'order': order,
                                            'editable': editable})
    return jsonschema_element

def jsonschema_sequence_recursion(value, title=None, order=None, editable=False):
    if title is None:
        title = 'Items'
    jsonschema_element = {'type': 'array', 'editable' : editable, 'title' : title}

    items_type = value.__args__[0]
    if hasattr(items_type, '_name') and items_type._name in ['List', 'Sequence', 'Iterable']:
        jsonschema_element['items'] = jsonschema_sequence_recursion(value=items_type,
                                                                    title=title,
                                                                    editable=editable)
    else:
        annotation = ('items', items_type)
        jsonschema_element.update(jsonschema_from_annotation(annotation,
                                                             jsonschema_element,
                                                             order=0,
                                                             title=title))
    return jsonschema_element

def prettyname(namestr):
    pretty_name = ''
    if namestr:
        strings = namestr.split('_')
        for i, string in enumerate(strings):
            if len(string) > 1:
                pretty_name += string[0].upper() + string[1:]
            else:
                pretty_name += string
            if i < len(strings)-1:
                pretty_name += ' '
    return pretty_name

def jsonschema_from_dataclass(class_, title=None):
    if title is None:
        title = prettyname(class_.__name__)
    jsonschema_element = {'type': 'object',
                          'properties' : {}}
    for i, field in enumerate(class_.__dataclass_fields__.values()): # !!! Not actually ordered !
        annotation = (field.name, field.type)
        jsonschema_element['properties'].update(jsonschema_from_annotation(annotation,
                                                                           jsonschema_element['properties'],
                                                                           order=i,
                                                                           title=title))
    return jsonschema_element

def set_default_value(jsonschema_element, key, default_value):
    if isinstance(default_value, tuple(TYPING_EQUIVALENCES.keys()))\
    or default_value is None:
        jsonschema_element[key]['default_value'] = default_value
    elif isinstance(default_value, (list, tuple)):
        raise NotImplementedError('List as default values not implemented')
    else:
        object_dict = default_value.to_dict()
        jsonschema_element[key]['default_value'] = object_dict
    return jsonschema_element



def inspect_arguments(method, merge=False):
    # Find default value and required arguments of class construction
    args_specs = inspect.getfullargspec(method)
    nargs = len(args_specs.args) - 1

    if args_specs.defaults is not None:
        ndefault_args = len(args_specs.defaults)
    else:
        ndefault_args = 0

    default_arguments = {}
    arguments = []
    for iargument, argument in enumerate(args_specs.args[1:]):
        if argument not in _FORBIDDEN_ARGNAMES:
            if iargument >= nargs - ndefault_args:
                default_value = args_specs.defaults[ndefault_args-nargs+iargument]
                if merge:
                    arguments.append((argument, default_value))
                else:
                    default_arguments[argument] = default_value
            else:
                arguments.append(argument)
    return arguments, default_arguments

def deserialize_argument(type_, argument):
    if isinstance(type_, TypeVar):
        # Get all classes
        classes = list(type_.__constraints__)
        instantiated = False
        while instantiated is False:
            # Find the last class in the hierarchy
            hierarchy_lengths = [len(cls.mro()) for cls in classes]
            children_class_index = hierarchy_lengths.index(max(hierarchy_lengths))
            children_class = classes[children_class_index]
            try:
                # Try to deserialize
                # Throws KeyError if we try to put wrong dict into dict_to_object
                # This means we try to instantiate a children class with a parent dict_to_object
                deserialized_argument = children_class.dict_to_object(argument)

                # If it succeeds we have the right class and instantiated object
                instantiated = True
            except KeyError:
                # This is not the right class, we should go see the parent
                classes.remove(children_class)
    elif hasattr(type_, '_name') and type_._name in ['List', 'Sequence', 'Iterable']:
        sequence_subtype = type_.__args__[0]
        deserialized_argument = [deserialize_argument(sequence_subtype, arg) for arg in argument]
    elif hasattr(type_, '_name') and type_._name == 'Dict':
        deserialized_argument = argument
    else:
        if type_ in TYPING_EQUIVALENCES.keys():
            if isinstance(argument, type_):
                deserialized_argument = argument
            else:
                if isinstance(argument, int) and type_ == float:
                    # explicit conversion in this case
                    deserialized_argument = float(argument)
                else:
                    raise TypeError('Given built-in type and argument are incompatible : {} and {} in {}'.format(type(argument), type_, argument))
        elif hasattr(type_, '__dataclass_fields__'):
            _ = type_(**argument)
            deserialized_argument = argument
        else:
            deserialized_argument = type_.dict_to_object(argument)
    return deserialized_argument


def recursive_type(obj):
    if isinstance(obj, tuple(list(TYPING_EQUIVALENCES.keys()) + [dict])):
        type_ = TYPES_STRINGS[type(obj)]
    elif isinstance(obj, DessiaObject):
        type_ = obj.__module__ + '.' + obj.__class__.__name__
    elif isinstance(obj, (list, tuple)):
        type_ = []
        for element in obj:
            type_.append(recursive_type(element))
    elif obj is None:
        type_ = None
    else:
        raise NotImplementedError(obj)
    return type_

def recursive_instantiation(types, values):
    instantiated_values = []
    for type_, value in zip(types, values):
        if type_ in TYPES_STRINGS.values():
            instantiated_values.append(eval(type_)(value))
#        if type_ in list(TYPING_EQUIVALENCES.keys()) + [dict, None]:
#            instantiated_values.append(value)
        elif isinstance(type_, str):
            exec('import ' + type_.split('.')[0])
            class_ = eval(type_)
            if inspect.isclass(class_):
                instantiated_values.append(class_.dict_to_object(value))
            else:
                raise NotImplementedError
        elif isinstance(type_, (list, tuple)):
            instantiated_values.append(recursive_instantiation(type_, value))
        elif type_ is None:
            instantiated_values.append(value)
        else:
            raise NotImplementedError(type_)
    return instantiated_values



JSONSCHEMA_HEADER = {"definitions": {},
                     "$schema": "http://json-schema.org/draft-07/schema#",
                     "type": "object",
                     "required" : [],
                     "properties": {}}

TYPING_EQUIVALENCES = {int: 'number',
                       float: 'number',
                       bool: 'boolean',
                       str: 'string'}

TYPES_STRINGS = {int: 'int', float: 'float', bool: 'boolean', str: 'str',
                 list: 'list', tuple: 'tuple', dict: 'dict'}

_FORBIDDEN_ARGNAMES = ['self', 'cls', 'progress_callback', 'return']
