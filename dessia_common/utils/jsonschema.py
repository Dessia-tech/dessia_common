#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" JsonSchema generation functions. """
from copy import deepcopy
import inspect
import warnings
import collections
import collections.abc
from typing import get_origin, get_args, Union, get_type_hints
from dessia_common.abstract import CoreDessiaObject
import dessia_common.utils.types as dc_types
from dessia_common.files import BinaryFile, StringFile
from dessia_common.typings import Subclass, MethodType, ClassMethodType, Any
from dessia_common.utils.docstrings import FAILED_ATTRIBUTE_PARSING
from dessia_common.measures import Measure
from dessia_common.utils.helpers import prettyname

JSONSCHEMA_HEADER = {"definitions": {},
                     "$schema": "http://json-schema.org/draft-07/schema#",
                     "type": "object",
                     "required": [],
                     "properties": {}}


def default_sequence(array_jsonschema):
    """ DEPRECATED. Soon to be removed. """
    if dc_types.is_sequence(array_jsonschema['items']):
        # Tuple jsonschema
        if 'default_value' in array_jsonschema:
            return array_jsonschema['default_value']
        return [default_dict(v) for v in array_jsonschema['items']]
    return None


def datatype_from_jsonschema(jsonschema):
    """ DEPRECATED. Soon to be removed. """
    if jsonschema['type'] == 'object':
        if 'classes' in jsonschema:
            if len(jsonschema['classes']) > 1:
                return 'union'
            if 'standalone_in_db' in jsonschema:
                if jsonschema['standalone_in_db']:
                    return 'standalone_object'
                return 'embedded_object'
            # Static dict is deprecated
            return 'static_dict'
        if 'instance_of' in jsonschema:
            return 'instance_of'
        if 'patternProperties' in jsonschema:
            return 'dynamic_dict'
        if 'is_method' in jsonschema and jsonschema['is_method']:
            return 'embedded_object'
        if 'is_class' in jsonschema and jsonschema['is_class']:
            return 'class'

    if jsonschema['type'] == 'array':
        if 'additionalItems' in jsonschema and not jsonschema['additionalItems']:
            return 'heterogeneous_sequence'
        return 'homogeneous_sequence'

    if jsonschema["type"] == "text" and "is_file" in jsonschema and jsonschema["is_file"]:
        return "file"

    if jsonschema['type'] in ['number', 'string', 'boolean']:
        return 'builtin'
    return None


def chose_default(jsonschema):
    """ DEPRECATED. Soon to be removed. """
    datatype = datatype_from_jsonschema(jsonschema)
    if datatype in ['heterogeneous_sequence', 'homogeneous_sequence']:
        return default_sequence(jsonschema)
    if datatype == 'static_dict':
        # Deprecated
        return default_dict(jsonschema)
    if datatype in ['standalone_object', 'embedded_object', 'instance_of', 'union']:
        if 'default_value' in jsonschema:
            return jsonschema['default_value']
        return None

    return None


def default_dict(jsonschema):
    """ DEPRECATED. Soon to be removed. """
    dict_ = {}
    datatype = datatype_from_jsonschema(jsonschema)
    if datatype in ['standalone_object', 'embedded_object', 'static_dict']:
        if 'classes' in jsonschema:
            dict_['object_class'] = jsonschema['classes'][0]
        elif 'is_method' in jsonschema and jsonschema['is_method']:
            # Method can have no classes in jsonschema
            pass
        else:
            raise ValueError(f"DessiaObject of type {jsonschema['python_typing']} must have 'classes' in jsonschema")
        for property_, jss in jsonschema['properties'].items():
            if 'default_value' in jss:
                dict_[property_] = jss['default_value']
            else:
                dict_[property_] = chose_default(jss)
    else:
        return None
    return dict_


def jsonschema_union_types(key, args, typing_, jsonschema_element):
    """ DEPRECATED. Soon to be removed. """
    classnames = [dc_types.full_classname(object_=a, compute_for='class') for a in args]
    standalone_args = [a._standalone_in_db for a in args]
    if all(standalone_args):
        standalone = True
    elif not any(standalone_args):
        standalone = False
    else:
        raise ValueError(f"standalone_in_db values for type '{typing_}' are not consistent")
    jsonschema_element[key].update({'type': 'object', 'classes': classnames, 'standalone_in_db': standalone})


def jsonschema_from_annotation(annotation, jsonschema_element, order, editable=None, title=None,
                               parsed_attributes=None):
    """ DEPRECATED. Soon to be removed. """
    key, typing_ = annotation
    if isinstance(typing_, str):
        raise ValueError

    if title is None:
        title = prettyname(key)
    if editable is None:
        editable = key not in ['return']

    if parsed_attributes is not None and key in parsed_attributes:
        try:
            description = parsed_attributes[key]['desc']
        except Exception:
            description = FAILED_ATTRIBUTE_PARSING["desc"]
    else:
        description = ""

    # Compute base entries
    jsonschema_element[key] = {'title': title, 'editable': editable, 'order': order, 'description': description,
                               'python_typing': dc_types.serialize_typing(typing_)}

    if typing_ in dc_types.TYPING_EQUIVALENCES:
        # Python Built-in type
        jsonschema_element[key]['type'] = dc_types.TYPING_EQUIVALENCES[typing_]

    elif dc_types.is_typing(typing_):
        origin = get_origin(typing_)
        args = get_args(typing_)
        if origin is Union:
            if dc_types.union_is_default_value(typing_):
                # This is a false Union => Is a default value set to None
                ann = (key, args[0])
                jsonschema_from_annotation(annotation=ann, jsonschema_element=jsonschema_element,
                                           order=order, editable=editable, title=title)
            else:
                # Types union
                jsonschema_union_types(key, args, typing_, jsonschema_element)
        elif origin in [list, collections.abc.Iterator]:
            # Homogeneous sequences
            jsonschema_element[key].update(jsonschema_sequence_recursion(value=typing_, order=order,
                                                                         title=title, editable=editable))
        elif origin is tuple:
            # Heterogenous sequences (tuples)
            items = []
            for type_ in args:
                items.append({'type': dc_types.TYPING_EQUIVALENCES[type_]})
            jsonschema_element[key].update({'additionalItems': False, 'type': 'array', 'items': items})
        elif origin is dict:
            # Dynamically created dict structure
            key_type, value_type = args
            if key_type != str:
                # !!! Should we support other types ? Numeric ?
                raise NotImplementedError('Non strings keys not supported')
            if value_type not in dc_types.TYPING_EQUIVALENCES:
                raise ValueError(f'Dicts should have only builtins keys and values, got {value_type}')
            jsonschema_element[key].update({'type': 'object',
                                            'patternProperties': {
                                                '.*': {
                                                    'type': dc_types.TYPING_EQUIVALENCES[value_type]
                                                }
                                            }})
        elif origin is Subclass:
            pass
            # warnings.simplefilter('once', DeprecationWarning)
            # msg = "\n\nTyping of attribute '{0}' from class {1} uses Subclass which is deprecated."\
            #       "\n\nUse 'InstanceOf[{2}]' instead of 'Subclass[{2}]'.\n"
            # arg = args[0].__name__
            # warnings.warn(msg.format(key, args[0], arg), DeprecationWarning)
            # # Several possible classes that are subclass of another one
            # class_ = args[0]
            # classname = dc.full_classname(object_=class_, compute_for='class')
            # jsonschema_element[key].update({'type': 'object', 'instance_of': classname,
            #                                 'standalone_in_db': class_._standalone_in_db})
        elif origin is dc_types.InstanceOf:
            # Several possible classes that are subclass of another one
            class_ = args[0]
            classname = dc_types.full_classname(object_=class_, compute_for='class')
            jsonschema_element[key].update({'type': 'object', 'instance_of': classname,
                                            'standalone_in_db': class_._standalone_in_db})
        elif origin is MethodType or origin is ClassMethodType:
            class_type = get_args(typing_)[0]
            classmethod_ = origin is ClassMethodType
            class_jss = jsonschema_from_annotation(annotation=('class_', class_type), jsonschema_element={},
                                                   order=order, editable=editable, title='Class')
            jsonschema_element[key].update({'type': 'object', 'is_method': True,
                                            'classmethod_': classmethod_,
                                            'properties': {
                                                'class_': class_jss['class_'],
                                                'name': {'type': 'string'}}})
        elif origin is type:
            jsonschema_element[key].update({'type': 'object', 'is_class': True,
                                            'properties': {'name': {'type': 'string'}}})
        else:
            raise NotImplementedError(f"Jsonschema computation of typing {typing_} is not implemented")

    elif hasattr(typing_, '__origin__') and typing_.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        jsonschema_element[key].update({'type': 'object', 'is_class': True,
                                        'properties': {'name': {'type': 'string'}}})
    elif typing_ is Any:
        jsonschema_element[key].update({'type': 'object', 'properties': {'.*': '.*'}})
    elif inspect.isclass(typing_) and issubclass(typing_, Measure):
        ann = (key, float)
        jsonschema_element = jsonschema_from_annotation(annotation=ann, jsonschema_element=jsonschema_element,
                                                        order=order, editable=editable, title=title)
        jsonschema_element[key]['units'] = typing_.si_unit
    elif inspect.isclass(typing_) and issubclass(typing_, (BinaryFile, StringFile)):
        jsonschema_element[key].update({'type': 'text', 'is_file': True})
    else:
        classname = dc_types.full_classname(object_=typing_, compute_for='class')
        if inspect.isclass(typing_) and issubclass(typing_, CoreDessiaObject):
            # Dessia custom classes
            jsonschema_element[key].update({'type': 'object', 'standalone_in_db': typing_._standalone_in_db})
        else:
            # DEPRECATED : Statically created dict structure
            jsonschema_element[key].update(static_dict_jsonschema(typing_))
        jsonschema_element[key]['classes'] = [classname]
    return jsonschema_element


def jsonschema_sequence_recursion(value, order: int, title: str = None,
                                  editable: bool = False):
    """ DEPRECATED. Soon to be removed. """
    if title is None:
        title = 'Items'
    jsonschema_element = {'type': 'array', 'order': order,
                          'python_typing': dc_types.serialize_typing(value)}

    items_type = get_args(value)[0]
    if dc_types.is_typing(items_type) and get_origin(items_type) is list:
        jss = jsonschema_sequence_recursion(value=items_type, order=0,
                                            title=title, editable=editable)
        jsonschema_element['items'] = jss
    else:
        annotation = ('items', items_type)
        jss = jsonschema_from_annotation(annotation=annotation, jsonschema_element=jsonschema_element,
                                         order=0, title=title)
        jsonschema_element.update(jss)
    return jsonschema_element


def static_dict_jsonschema(typed_dict, title=None):
    """ DEPRECATED. Soon to be removed. """
    warnings.simplefilter('once', DeprecationWarning)
    msg = "\n\nStatic Dict typing is not fully supported.\n" \
          "This will most likely lead to non predictable behavior" \
          " or malfunctionning features. \n" \
          "Define a custom non-standalone class for type '{}'\n\n"
    classname = dc_types.full_classname(typed_dict, compute_for='class')
    warnings.warn(msg.format(classname), DeprecationWarning)
    jsonschema_element = deepcopy(JSONSCHEMA_HEADER)
    jss_properties = jsonschema_element['properties']

    # Every value is required in a StaticDict
    annotations = get_type_hints(typed_dict)
    jsonschema_element['required'] = list(annotations.keys())

    # TOCHECK : Not actually ordered !
    for i, annotation in enumerate(annotations.items()):
        attribute, typing_ = annotation
        if not dc_types.is_builtin(typing_):
            msg = "Complex structure as Static dict values is not supported." \
                  "\n Attribute {} got type {} instead of builtin." \
                  "\n Consider creating a custom class for complex structures."
            raise TypeError(msg.format(attribute, typing_))
        jss = jsonschema_from_annotation(annotation=annotation,
                                         jsonschema_element=jss_properties,
                                         order=i, title=title)
        jss_properties.update(jss)
    return jsonschema_element


def set_default_value(jsonschema_element, key, default_value):
    """ DEPRECATED. Soon to be removed. """
    datatype = datatype_from_jsonschema(jsonschema_element[key])
    if default_value is None\
            or datatype in ['builtin', 'heterogeneous_sequence',
                            'static_dict', 'dynamic_dict']:
        jsonschema_element[key]['default_value'] = default_value
    # elif datatype == 'builtin':
    #     jsonschema_element[key]['default_value'] = default_value
    # elif datatype == 'heterogeneous_sequence':
    #     jsonschema_element[key]['default_value'] = default_value
    elif datatype == 'homogeneous_sequence':
        msg = 'Object {} of type {} is not supported as default value'
        type_ = type(default_value)
        raise NotImplementedError(msg.format(default_value, type_))
    elif datatype in ['standalone_object', 'embedded_object',
                      'instance_of', 'union']:
        object_dict = default_value.to_dict(use_pointers=False)
        jsonschema_element[key]['default_value'] = object_dict
    return jsonschema_element
    # if isinstance(default_value, tuple(TYPING_EQUIVALENCES.keys())) \
    #         or default_value is None:
    #     jsonschema_element[key]['default_value'] = default_value
    # elif is_sequence(default_value):
    #     if datatype == 'heterogeneous_sequence':
    #         jsonschema_element[key]['default_value'] = default_value
    #     else:
    #         msg = 'Object {} of type {} is not supported as default value'
    #         type_ = type(default_value)
    #         raise NotImplementedError(msg.format(default_value, type_))
    # else:
    #     if datatype in ['standalone_object', 'embedded_object',
    #                     'subclass', 'union']:
    #     object_dict = default_value.to_dict()
    #     jsonschema_element[key]['default_value'] = object_dict
    #     else:
