#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" JsonSchema generation functions. """
import warnings
import dessia_common.utils.types as dc_types

JSONSCHEMA_HEADER = {"definitions": {},
                     "$schema": "http://json-schema.org/draft-07/schema#",
                     "type": "object",
                     "required": [],
                     "properties": {}}

TYPING_EQUIVALENCES = {int: 'number', float: 'number', bool: 'boolean', str: 'string'}


def default_sequence(array_jsonschema):
    """ DEPRECATED. Soon to be removed. """
    warnings.warn("Jsonschema module is deprecated and will be removed soon. Use schemas instead.", DeprecationWarning)
    if dc_types.is_sequence(array_jsonschema['items']):
        # Tuple jsonschema
        if 'default_value' in array_jsonschema:
            return array_jsonschema['default_value']
        return [default_dict(v) for v in array_jsonschema['items']]
    return None


def datatype_from_jsonschema(jsonschema):
    """ DEPRECATED. Soon to be removed. """
    warnings.warn("Jsonschema module is deprecated and will be removed soon. Use schemas instead.", DeprecationWarning)
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
    warnings.warn("Jsonschema module is deprecated and will be removed soon. Use schemas instead.", DeprecationWarning)
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
    warnings.warn("Jsonschema module is deprecated and will be removed soon. Use schemas instead.", DeprecationWarning)
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

