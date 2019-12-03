#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import inspect
from copy import deepcopy

from typing import TypeVar

class DessiaObject:
    """

    """

    @classmethod
    def base_jsonschema(cls):
        jsonschema = JSONSCHEMA_HEADER.copy()
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
#            _jsonschema = dict_merge(DessiaObject.base_jsonschema(), cls._jsonschema)
            _jsonschema = cls._jsonschema
            return _jsonschema

        # Get __init__ method and its annotations
        init = cls.__init__
        annotations = init.__annotations__

#        print(annotations)

        # Get editable and ordered variables
        if hasattr(cls, '_editable_variables') and cls._editable_variables is not None:
            editable_variables = cls._editable_variables
        else:
            editable_variables = list(annotations.keys())

        if hasattr(cls, '_ordered_variables') and cls._ordered_variables is not None:
            ordered_variables = cls._ordered_variables
        else:
            ordered_variables = editable_variables

        if hasattr(cls, '_titled_variables'):
            titled_variables = cls._titled_variables
        else:
            titled_variables = None
        unordered_count = 0

        # Initialize jsonschema
        _jsonschema = deepcopy(JSONSCHEMA_HEADER)

        required_arguments, default_arguments = inspect_arguments(init, merge=False)
        _jsonschema['required'] = required_arguments

        # Set jsonschema
        for annotation in annotations.items():
            name, value = annotation
            if name in ordered_variables:
                order = ordered_variables.index(name)
            else:
                order = len(ordered_variables) + unordered_count
                unordered_count += 1
            if titled_variables is not None and name in titled_variables:
                title = titled_variables[name]
            else:
                title = None
            jsonschema_element = jsonschema_from_annotation(annotation=annotation,
                                                            jsonschema_element={},
                                                            order=order,
                                                            editable=name in editable_variables,
                                                            title=title)
            _jsonschema['properties'].update(jsonschema_element)
            if name in default_arguments.keys():
                default = set_default_value(_jsonschema['properties'],
                                            name,
                                            default_arguments[name])
                _jsonschema['properties'].update(default)
        return _jsonschema

    @property
    def _method_jsonschemas(self):
        """
        Generates dynamic jsonschemas for methods of class
        """
        jsonschemas = {}
        cls = type(self)
        valid_method_names = [m for m in dir(cls)\
                              if not m.startswith('_')]
        for method_name in valid_method_names:
            method = getattr(cls, method_name)

            required_arguments, default_arguments = inspect_arguments(method, merge=False)

            if method.__annotations__:
                jsonschemas[method_name] = deepcopy(JSONSCHEMA_HEADER)
                jsonschemas[method_name]['required'] = required_arguments
                for i, annotation in enumerate(method.__annotations__.items()): # !!! Not actually ordered
                    jsonschema_element = jsonschema_from_annotation(annotation, {}, i)[annotation[0]]
                    jsonschemas[method_name]['properties'][str(i)] = jsonschema_element
                    if annotation[0] in default_arguments.keys():
                        default = set_default_value(jsonschemas[method_name]['properties'],
                                                    str(i),
                                                    default_arguments[annotation[0]])
                        jsonschemas[method_name]['properties'].update(default)
        return jsonschemas

    def base_dict_to_arguments(self, dict_, method):
        method_object = getattr(self, method)
        args_specs = inspect.getfullargspec(method_object)
        allowed_args = args_specs.args
        self_index = allowed_args.index('self')
        allowed_args.pop(self_index)

        arguments = {}
        for i, arg in enumerate(allowed_args):
            if str(i) in dict_:
                value = dict_[str(i)]
                if hasattr(value, 'to_dict'):
                    serialized_value = value.to_dict()
                else:
                    serialized_value = value
                arguments[arg] = serialized_value
        return arguments

    
def jsonschema_from_annotation(annotation, jsonschema_element,
                               order, editable=None, title=None):
    key, value = annotation
    if title is None:
        title = prettyname(key)
    if editable is None:
        editable = key not in ['return']
    if value in TYPING_EQUIVALENCES.keys():
        # Python Built-in type
        jsonschema_element[key] = {'type': TYPING_EQUIVALENCES[value],
                                   'title': title,
                                   'editable' : editable,
                                   'order' : order}
    elif isinstance(value, TypeVar):
        # Several  classes are possible
        classnames = [c.__module__+'.'+c.__name__ for c in value.__constraints__]
        jsonschema_element[key] = {'type': 'object',
                                   'classes': classnames,
                                   'title': title,
                                   'editable' : editable,
                                   'order' : order}
    elif hasattr(value, '_name')\
    and value._name in ['List', 'Sequence', 'Iterable']:
        items_type = value.__args__[0]
        if items_type in TYPING_EQUIVALENCES.keys():
            jsonschema_element[key] = {'type': 'array',
                                       'title': title,
                                       'order' : order,
                                       'editable' : editable,
                                       'items': {
                                           'type': TYPING_EQUIVALENCES[items_type]
                                           }
                                       }
        elif hasattr(value, '_standalone_in_db') or not hasattr(items_type, '__dataclass_fields__'):
            classname = items_type.__module__ + '.' + items_type.__name__
            # List of a certain type
            jsonschema_element[key] = {'type': 'array',
                                       'title': title,
                                       'order' : order,
                                       'editable' : editable,
                                       'items': {
                                           'type': 'object',
                                           'classes': [classname]
                                           }
                                       }
        else:
            jsonschema_element[key] = {'type': 'array',
                                       'title': title,
                                       'order' : order,
                                       'editable' : editable,
                                       'items': jsonschema_from_dataclass(items_type)}

    else:
        if hasattr(value, '_standalone_in_db'):
            # Dessia custom classes
            classname = value.__module__ + '.' + value.__name__
            jsonschema_element[key] = {'type': 'object',
                                       'title': title,
                                       'order' : order,
                                       'editable' : editable,
                                       'classes': [classname]}
        else:
            # Dataclasses
            jsonschema_element[key] = jsonschema_from_dataclass(value)
            jsonschema_element[key]['title'] = title
            jsonschema_element[key]['order'] = order
            jsonschema_element[key]['editable'] = editable

    return jsonschema_element

def prettyname(namestr):
    prettyname = ''
    if namestr:
        strings = namestr.split('_')
        for i, string in enumerate(strings):
            prettyname += string[0].upper() + string[1:]
            if i < len(strings)-1:
                prettyname += ' '
    return prettyname

def jsonschema_from_dataclass(class_):
    jsonschema_element = {'type': 'object',
                          'properties' : {}}
    for i, field in enumerate(class_.__dataclass_fields__.values()): # !!! Not actually ordered !
        if field.type in TYPING_EQUIVALENCES.keys():
            current_dict = {'type': TYPING_EQUIVALENCES[field.type],
                            'title': prettyname(field.name),
                            'order': i,
                            'editable': True} # !!! Dynamic editable field ?
        else:
            current_dict = jsonschema_from_dataclass(field.type)
            current_dict['order'] = i
            current_dict['editable'] = True # !!! Dynamic editable field ?
        jsonschema_element['properties'][field.name] = current_dict
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
        if not argument in ['self', 'progress_callback']:
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
    elif hasattr(type_, '_name')\
    and type_._name in ['List', 'Sequence', 'Iterable']:
        sequence_subtype = type_.__args__[0]
        deserialized_argument = [deserialize_argument(sequence_subtype, arg) for arg in argument]
    else:
         if type_ in TYPING_EQUIVALENCES.keys():
             if isinstance(argument, type_):
                 deserialized_argument = argument
             else:
                 raise TypeError('Given built-in type and argument are incompatible : {} and {}'.format(type(argument), type_))
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
        print(obj)
        raise NotImplementedError
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
            print(type_)
            raise NotImplementedError
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
