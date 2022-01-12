#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:22:23 2022

@author: steven
"""

import dessia_common.utils.types as dc_types

def default_sequence(array_jsonschema):
    if dc_types.is_sequence(array_jsonschema['items']):
        # Tuple jsonschema
        if 'default_value' in array_jsonschema:
            return array_jsonschema['default_value']
        return [default_dict(v) for v in array_jsonschema['items']]
    return None


def datatype_from_jsonschema(jsonschema):
    if jsonschema['type'] == 'object':
        if 'classes' in jsonschema:
            if len(jsonschema['classes']) > 1:
                return 'union'
            if 'standalone_in_db' in jsonschema:
                if jsonschema['standalone_in_db']:
                    return 'standalone_object'
                return 'embedded_object'
            return 'static_dict'
        if 'instance_of' in jsonschema:
            return 'instance_of'
        if 'patternProperties' in jsonschema:
            return 'dynamic_dict'
        if 'method' in jsonschema and jsonschema['method']:
            return 'embedded_object'
        if 'is_class' in jsonschema and jsonschema['is_class']:
            return 'class'

    elif jsonschema['type'] == 'array':
        if 'additionalItems' in jsonschema\
                and not jsonschema['additionalItems']:
            return 'heterogeneous_sequence'
        return 'homogeneous_sequence'

    elif jsonschema['type'] in ['number', 'string', 'boolean']:
        if 'is_type' in jsonschema and jsonschema['is_type']:
            return 'file'
        return 'builtin'
    return None


def chose_default(jsonschema):
    datatype = datatype_from_jsonschema(jsonschema)
    if datatype in ['heterogeneous_sequence', 'homogeneous_sequence']:
        return default_sequence(jsonschema)
    elif datatype == 'static_dict':
        return default_dict(jsonschema)
    elif datatype in ['standalone_object', 'embedded_object',
                      'instance_of', 'union']:
        if 'default_value' in jsonschema:
            return jsonschema['default_value']
        return None
    else:
        return None


def default_dict(jsonschema):
    dict_ = {}
    datatype = datatype_from_jsonschema(jsonschema)
    if datatype in ['standalone_object', 'embedded_object', 'static_dict']:
        if 'classes' in jsonschema:
            dict_['object_class'] = jsonschema['classes'][0]
        elif 'method' in jsonschema and jsonschema['method']:
            # Method can have no classes in jsonschema
            pass
        else:
            msg = "DessiaObject of type {} must have 'classes' in jsonschema"
            raise ValueError(msg.format(jsonschema['python_typing']))
        for property_, jss in jsonschema['properties'].items():
            if 'default_value' in jss:
                dict_[property_] = jss['default_value']
            else:
                dict_[property_] = chose_default(jss)
    else:
        return None
    return dict_
