#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JsonSchema generation functions
"""
from copy import deepcopy
import inspect
import collections
import collections.abc
from typing import get_origin, get_args, Union, get_type_hints, Type, Callable, List, Tuple, TypedDict
import dessia_common as dc
import dessia_common.utils.types as dc_types
from dessia_common.files import BinaryFile, StringFile
from dessia_common.typings import Measure, Subclass, MethodType, ClassMethodType, Any
from dessia_common.utils.docstrings import parse_docstring, FAILED_DOCSTRING_PARSING, FAILED_ATTRIBUTE_PARSING


JSONSCHEMA_HEADER = {"definitions": {},
                     "$schema": "http://json-schema.org/draft-07/schema#",
                     "type": "object",
                     "required": [],
                     "properties": {}}

Issue = TypedDict("Issue", {"attribute": str, "severity": str, "message": str})


class Jsonschema:
    _untreated_argnames = ["self", "return"]

    def __init__(self, annotations, argspec, docstring):
        self.annotations = annotations
        self.attributes = [a for a in argspec.args if a not in self._untreated_argnames]

        self.standalone_in_db = None
        self.python_typing = ""

        # Parse docstring
        try:
            self.parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        except Exception:
            self.parsed_docstring = FAILED_DOCSTRING_PARSING

        self.parsed_attributes = self.parsed_docstring['attributes']

        self.required_arguments, self.default_arguments = dc.inspect_arguments(argspec=argspec, merge=False)

    def annotations_are_valid(self) -> Tuple[bool, List[Issue]]:
        issues = []
        for attribute in self.attributes:
            annotation = self.annotations[attribute]
            if dc_types.is_typing(annotation):
                origin = get_origin(annotation)
                args = get_args(annotation)
                if origin is dict and len(args) != 2:
                    msg = f"Attribute '{attribute}' is typed as a 'Dict' which requires exactly 2 arguments." \
                          f"Expected Dict[KeyType, ValueType], got {annotation}."
                    issues.append({"attribute": attribute, "severity": "error", "message": msg})
                if origin is list and len(args) != 1:
                    msg = f"Attribute '{attribute}' is typed as a 'List' which requires exactly 1 argument." \
                          f"Expected List[Type], got {annotation}."
                    issues.append({"attribute": attribute, "severity": "error", "message": msg})
                if origin is tuple and len(args) == 0:
                    msg = f"Attribute '{attribute}' is typed as a 'Tuple' which requires at least 1 argument." \
                          f"Expected Tuple[Type0, Type1, ..., TypeN], got {annotation}."
                    issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return not any(issues), issues

    def chunk(self, attribute):
        annotation = self.annotations[attribute]

        if self.parsed_attributes is not None and attribute in self.parsed_attributes:
            try:
                description = self.parsed_attributes[attribute]['desc']
            except Exception:
                description = FAILED_ATTRIBUTE_PARSING["desc"]
        else:
            description = ""
        chunk = jsonschema_chunk(annotation=annotation, title=dc.prettyname(attribute),
                                 editable=attribute not in ['return'], description=description)
        if attribute in self.default_arguments:
            chunk = set_default_value(jsonschema_element=chunk, default_value=self.default_arguments[attribute])
        return chunk

    @property
    def chunks(self):
        return [self.chunk(a) for a in self.attributes]

    def write(self):
        jsonschema = deepcopy(JSONSCHEMA_HEADER)
        properties = {a: self.chunk(a) for a in self.attributes}
        jsonschema.update({"required": self.required_arguments, "properties": properties,
                           "description": self.parsed_docstring["description"]})
        return jsonschema


class ClassJsonschema(Jsonschema):
    def __init__(self, class_: Type):
        self.class_ = class_
        self.standalone_in_db = class_._standalone_in_db
        self.python_typing = str(class_)
        annotations = get_type_hints(class_.__init__)

        members = inspect.getfullargspec(self.class_.__init__)
        docstring = class_.__doc__

        Jsonschema.__init__(self, annotations=annotations, argspec=members, docstring=docstring)

    def check(self) -> Tuple[bool, List[Issue]]:
        issues = []
        for attribute in self.attributes:
            if attribute not in self.annotations:
                msg = f"Property {attribute} has no typing"
                issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return not any(issues), issues


class MethodJsonschema(Jsonschema):
    def __init__(self, method: Callable):
        self.method = method

        annotations = get_type_hints(method)
        Jsonschema.__init__(self, annotations=annotations)


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


def jsonschema_union_types(annotation):
    args = get_args(annotation)
    classnames = [dc.full_classname(object_=a, compute_for='class') for a in args]
    standalone_args = [a._standalone_in_db for a in args]
    if all(standalone_args):
        standalone = True
    elif not any(standalone_args):
        standalone = False
    else:
        raise ValueError(f"standalone_in_db values for type '{annotation}' are not consistent")
    return {'type': 'object', 'classes': classnames, 'standalone_in_db': standalone}


def jsonschema_from_annotation(annotation, jsonschema_element, order, editable=None, title=None,
                               parsed_attributes=None):
    key, typing_ = annotation
    if isinstance(typing_, str):
        raise ValueError

    if title is None:
        title = dc.prettyname(key)
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
            # Homogenous sequences
            jsonschema_element[key].update(jsonschema_sequence_recursion(value=typing_, order=order,
                                                                         title=title, editable=editable))
        elif origin is tuple:
            # Heterogenous sequences (tuples)
            jsonschema_element[key].update(tuple_jsonschema(args))
        elif origin is dict:
            # Dynamically created dict structure
            jsonschema_element[key].update(dynamic_dict_jsonschema(args))
        elif origin is Subclass:
            pass
            # warnings.simplefilter('once', DeprecationWarning)
            # msg = "\n\nTyping of attribute '{0}' from class {1} uses Subclass which is deprecated."\
            #       "\n\nUse 'InstanceOf[{2}]' instead of 'Subclass[{2}]'.\n"
            # arg = args[0].__name__
            # warnings.warn(msg.format(key, args[0], arg), DeprecationWarning)
            # # Several possible classes that are subclass of another one
            # class_ = args[0]
            # classname = full_classname(object_=class_, compute_for='class')
            # jsonschema_element[key].update({'type': 'object', 'instance_of': classname,
            #                                 'standalone_in_db': class_._standalone_in_db})
        elif origin is dc_types.InstanceOf:
            # Several possible classes that are subclass of another one
            jsonschema_element[key].update(instance_of_jsonschema(args))
        elif origin is MethodType or origin is ClassMethodType:
            jsonschema_element[key].update(method_type_jsonschema(typing_=typing_, order=order, editable=editable))
        elif origin is type:
            jsonschema_element[key].update(class_jsonschema)
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
        jsonschema_element[key]['units'] = typing_.units
    elif inspect.isclass(typing_) and issubclass(typing_, (BinaryFile, StringFile)):
        jsonschema_element[key].update({'type': 'text', 'is_file': True})
    else:
        classname = dc.full_classname(object_=typing_, compute_for='class')
        if inspect.isclass(typing_) and issubclass(typing_, dc.DessiaObject):
            # Dessia custom classes
            jsonschema_element[key].update({'type': 'object', 'standalone_in_db': typing_._standalone_in_db})
        # else:
            # DEPRECATED : Statically created dict structure
            # jsonschema_element[key].update(static_dict_jsonschema(typing_))
        jsonschema_element[key]['classes'] = [classname]
    return jsonschema_element


def jsonschema_chunk(annotation, title: str, editable: bool, description: str):
    if isinstance(annotation, str):
        raise ValueError

    if annotation in dc_types.TYPING_EQUIVALENCES:
        # Python Built-in type
        chunk = builtin_jsonschema(annotation)
    elif dc_types.is_typing(annotation):
        chunk = typing_jsonschema(typing_=annotation, title=title, editable=editable, description=description)
    elif hasattr(annotation, '__origin__') and annotation.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        chunk = {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
    elif annotation is Any:
        chunk = {'type': 'object', 'properties': {'.*': '.*'}}
    elif inspect.isclass(annotation) and issubclass(annotation, Measure):
        chunk = jsonschema_chunk(annotation=float, title=title, editable=editable, description=description)
        chunk['units'] = annotation.units
    elif inspect.isclass(annotation) and issubclass(annotation, (BinaryFile, StringFile)):
        chunk = {'type': 'text', 'is_file': True}
    elif inspect.isclass(annotation) and issubclass(annotation, dc.DessiaObject):
        # Dessia custom classes
        classname = dc.full_classname(object_=annotation, compute_for='class')
        chunk = {'type': 'object', 'standalone_in_db': annotation._standalone_in_db, "classes": [classname]}
    else:
        raise NotImplementedError
    chunk.update({'title': title, 'editable': editable, 'description': description,
                  'python_typing': dc_types.serialize_typing(annotation)})
    return chunk


def typing_jsonschema(typing_, title: str, editable: bool, description: str):
    origin = get_origin(typing_)
    if origin is Union:
        if dc_types.union_is_default_value(typing_):
            # This is a false Union => Is a default value set to None
            return jsonschema_chunk(annotation=typing_, title=title, editable=editable, description=description)
        # Types union
        return jsonschema_union_types(typing_)
    if origin in [list, collections.abc.Iterator]:
        # Homogenous sequences
        return jsonschema_sequence_recursion(value=typing_, title=title, editable=editable)
    if origin is tuple:
        # Heterogenous sequences (tuples)
        return tuple_jsonschema(typing_)
    if origin is dict:
        # Dynamically created dict structure)
        return dynamic_dict_jsonschema(typing_)
    if origin is Subclass:
        pass
        # warnings.simplefilter('once', DeprecationWarning)
        # msg = "\n\nTyping of attribute '{0}' from class {1} uses Subclass which is deprecated."\
        #       "\n\nUse 'InstanceOf[{2}]' instead of 'Subclass[{2}]'.\n"
        # arg = args[0].__name__
        # warnings.warn(msg.format(key, args[0], arg), DeprecationWarning)
        # # Several possible classes that are subclass of another one
        # class_ = args[0]
        # classname = full_classname(object_=class_, compute_for='class')
        # jsonschema_element[key].update({'type': 'object', 'instance_of': classname,
        #                                 'standalone_in_db': class_._standalone_in_db})
    if origin is dc_types.InstanceOf:
        # Several possible classes that are subclass of another one
        return instance_of_jsonschema(typing_)
    if origin is MethodType or origin is ClassMethodType:
        return method_type_jsonschema(annotation=typing_, editable=editable)
    if origin is type:
        return class_jsonschema()
    raise NotImplementedError(f"Jsonschema computation of typing {typing_} is not implemented")


def jsonschema_sequence_recursion(value, title: str = None, editable: bool = False):
    if title is None:
        title = 'Items'
    chunk = {'type': 'array', 'python_typing': dc_types.serialize_typing(value)}

    items_type = get_args(value)[0]
    if dc_types.is_typing(items_type) and get_origin(items_type) is list:
        chunk['items'] = jsonschema_sequence_recursion(value=items_type, title=title, editable=editable)
    else:
        chunk["items"] = jsonschema_chunk(annotation=items_type, title=title, editable=editable, description="")
    return chunk


def set_default_value(jsonschema_element, default_value):
    datatype = datatype_from_jsonschema(jsonschema_element)
    if default_value is None or datatype in ['builtin', 'heterogeneous_sequence', 'static_dict', 'dynamic_dict']:
        jsonschema_element['default_value'] = default_value
    # elif datatype == 'builtin':
    #     jsonschema_element[key]['default_value'] = default_value
    # elif datatype == 'heterogeneous_sequence':
    #     jsonschema_element[key]['default_value'] = default_value
    elif datatype == 'homogeneous_sequence':
        msg = 'Object {} of type {} is not supported as default value'
        type_ = type(default_value)
        raise NotImplementedError(msg.format(default_value, type_))
    elif datatype in ['standalone_object', 'embedded_object', 'instance_of', 'union']:
        object_dict = default_value.to_dict()
        jsonschema_element['default_value'] = object_dict
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


def builtin_jsonschema(annotation):
    chunk = {"type": dc_types.TYPING_EQUIVALENCES[annotation], 'python_typing': dc_types.serialize_typing(annotation)}
    return chunk


def tuple_jsonschema(annotation):
    args = get_args(annotation)
    items = []
    for type_ in args:
        items.append({'type': dc_types.TYPING_EQUIVALENCES[type_]})
    return {'additionalItems': False, 'type': 'array', 'items': items}


def dynamic_dict_jsonschema(annotation):
    args = get_args(annotation)
    key_type, value_type = args
    if key_type != str:
        # !!! Should we support other types ? Numeric ?
        raise NotImplementedError('Non strings keys not supported')
    if value_type not in dc_types.TYPING_EQUIVALENCES:
        raise ValueError(f'Dicts should have only builtins keys and values, got {value_type}')
    jsonschema = {
        'type': 'object',
        'patternProperties': {
            '.*': {
                'type': dc_types.TYPING_EQUIVALENCES[value_type]
            }
        }
    }
    return jsonschema


def instance_of_jsonschema(annotation):
    args = get_args(annotation)
    class_ = args[0]
    classname = dc.full_classname(object_=class_, compute_for='class')
    return {'type': 'object', 'instance_of': classname, 'standalone_in_db': class_._standalone_in_db}


def method_type_jsonschema(annotation, editable):
    origin = get_origin(annotation)
    class_type = get_args(annotation)[0]
    classmethod_ = origin is ClassMethodType
    chunk = jsonschema_chunk(annotation=class_type, title="Class", editable=editable, description="")
    jsonschema = {
        'type': 'object', 'is_method': True, 'classmethod_': classmethod_,
        'properties': {
            'class_': chunk,
            'name': {
                'type': 'string'
            }
        }
    }
    return jsonschema


def class_jsonschema():
    return {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
