#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema generation functions
"""
from copy import deepcopy
import inspect
import collections
import collections.abc
from typing import get_origin, get_args, Union, get_type_hints, Type, Callable, List, Tuple, TypedDict
import dessia_common as dc
import dessia_common.utils.types as dc_types
from dessia_common.files import BinaryFile, StringFile
from dessia_common.utils.docstrings import parse_docstring, FAILED_DOCSTRING_PARSING, FAILED_ATTRIBUTE_PARSING
from dessia_common.typings import Subclass, MethodType, ClassMethodType, Any
from dessia_common.measures import Measure

SCHEMA_HEADER = {"definitions": {},
                 "$schema": "http://json-schema.org/draft-07/schema#",
                 "type": "object",
                 "required": [],
                 "properties": {}}

Issue = TypedDict("Issue", {"attribute": str, "severity": str, "message": str})


class Schema:
    _untreated_argnames = ["self", "return"]

    def __init__(self, annotations, argspec, docstring):
        self.annotations = annotations
        self.attributes = [a for a in argspec.args if a not in self._untreated_argnames]

        self.python_typing = ""

        # Parse docstring
        try:
            self.parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        except Exception:
            self.parsed_docstring = FAILED_DOCSTRING_PARSING

        self.parsed_attributes = self.parsed_docstring['attributes']

        self.required_arguments, self.default_arguments = dc.split_default_args(argspec=argspec, merge=False)

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
        chunk = schema_chunk(annotation=annotation, title=dc.prettyname(attribute),
                             editable=attribute not in ['return'], description=description)
        if attribute in self.default_arguments:
            chunk = set_default_value(schema_element=chunk, default_value=self.default_arguments[attribute])
        return chunk

    @property
    def chunks(self):
        return [self.chunk(a) for a in self.attributes]

    def write(self):
        schema = deepcopy(SCHEMA_HEADER)
        properties = {a: self.chunk(a) for a in self.attributes}
        schema.update({"required": self.required_arguments, "properties": properties,
                       "description": self.parsed_docstring["description"]})
        return schema


class ClassSchema(Schema):
    def __init__(self, class_: Type):
        self.class_ = class_
        self.standalone_in_db = class_._standalone_in_db
        self.python_typing = str(class_)
        annotations = get_type_hints(class_.__init__)

        members = inspect.getfullargspec(self.class_.__init__)
        docstring = class_.__doc__

        Schema.__init__(self, annotations=annotations, argspec=members, docstring=docstring)

    def check(self) -> Tuple[bool, List[Issue]]:
        issues = []
        for attribute in self.attributes:
            if attribute not in self.annotations:
                msg = f"Property {attribute} has no typing"
                issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return not any(issues), issues

    def write(self):
        schema = Schema.write(self)
        schema["standalone_in_db"] = self.standalone_in_db
        return schema


class MethodSchema(Schema):
    def __init__(self, method: Callable):
        self.method = method

        annotations = get_type_hints(method)
        Schema.__init__(self, annotations=annotations)


def default_sequence(array_schema):
    if dc_types.is_sequence(array_schema['items']):
        # Tuple schema
        if 'default_value' in array_schema:
            return array_schema['default_value']
        return [default_dict(v) for v in array_schema['items']]
    return None


def datatype_from_schema(schema):
    if schema['type'] == 'object':
        if 'classes' in schema:
            if len(schema['classes']) > 1:
                return 'union'
            if 'standalone_in_db' in schema:
                if schema['standalone_in_db']:
                    return 'standalone_object'
                return 'embedded_object'
            # Static dict is deprecated
            return 'static_dict'
        if 'instance_of' in schema:
            return 'instance_of'
        if 'patternProperties' in schema:
            return 'dynamic_dict'
        if 'is_method' in schema and schema['is_method']:
            return 'embedded_object'
        if 'is_class' in schema and schema['is_class']:
            return 'class'

    if schema['type'] == 'array':
        if 'additionalItems' in schema and not schema['additionalItems']:
            return 'heterogeneous_sequence'
        return 'homogeneous_sequence'

    if schema["type"] == "text" and "is_file" in schema and schema["is_file"]:
        return "file"

    if schema['type'] in ['number', 'string', 'boolean']:
        return 'builtin'
    return None


def chose_default(schema):
    datatype = datatype_from_schema(schema)
    if datatype in ['heterogeneous_sequence', 'homogeneous_sequence']:
        return default_sequence(schema)
    if datatype == 'static_dict':
        # Deprecated
        return default_dict(schema)
    if datatype in ['standalone_object', 'embedded_object', 'instance_of', 'union']:
        if 'default_value' in schema:
            return schema['default_value']
        return None

    return None


def default_dict(schema):
    dict_ = {}
    datatype = datatype_from_schema(schema)
    if datatype in ['standalone_object', 'embedded_object', 'static_dict']:
        if 'classes' in schema:
            dict_['object_class'] = schema['classes'][0]
        elif 'is_method' in schema and schema['is_method']:
            # Method can have no classes in schema
            pass
        else:
            raise ValueError(f"DessiaObject of type {schema['python_typing']} must have 'classes' in schema")
        for property_, jss in schema['properties'].items():
            if 'default_value' in jss:
                dict_[property_] = jss['default_value']
            else:
                dict_[property_] = chose_default(jss)
    else:
        return None
    return dict_


def schema_union_types(annotation):
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


def schema_from_annotation(annotation, schema_element, order, editable=None,
                           title=None, parsed_attributes=None):
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
    schema_element[key] = {'title': title, 'editable': editable, 'order': order, 'description': description,
                           'python_typing': dc_types.serialize_typing(typing_)}

    if typing_ in dc_types.TYPING_EQUIVALENCES:
        # Python Built-in type
        schema_element[key]['type'] = dc_types.TYPING_EQUIVALENCES[typing_]

    elif dc_types.is_typing(typing_):
        origin = get_origin(typing_)
        args = get_args(typing_)
        if origin is Union:
            if dc_types.union_is_default_value(typing_):
                # This is a false Union => Is a default value set to None
                ann = (key, args[0])
                schema_from_annotation(annotation=ann, schema_element=schema_element,
                                       order=order, editable=editable, title=title)
            else:
                # Types union
                schema_union_types(key, args, typing_, schema_element)
        elif origin in [list, collections.abc.Iterator]:
            # Homogenous sequences
            schema_element[key].update(schema_sequence_recursion(value=typing_, order=order,
                                                                 title=title, editable=editable))
        elif origin is tuple:
            # Heterogenous sequences (tuples)
            schema_element[key].update(tuple_schema(args))
        elif origin is dict:
            # Dynamically created dict structure
            schema_element[key].update(dynamic_dict_schema(args))
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
            # schema_element[key].update({'type': 'object', 'instance_of': classname,
            #                                 'standalone_in_db': class_._standalone_in_db})
        elif origin is dc_types.InstanceOf:
            # Several possible classes that are subclass of another one
            schema_element[key].update(instance_of_schema(args))
        elif origin is MethodType or origin is ClassMethodType:
            schema_element[key].update(method_type_schema(typing_=typing_, order=order, editable=editable))
        elif origin is type:
            schema_element[key].update(class_schema)
        else:
            raise NotImplementedError(f"Schema computation of typing {typing_} is not implemented")

    elif hasattr(typing_, '__origin__') and typing_.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        schema_element[key].update({'type': 'object', 'is_class': True,
                                    'properties': {'name': {'type': 'string'}}})
    elif typing_ is Any:
        schema_element[key].update({'type': 'object', 'properties': {'.*': '.*'}})
    elif inspect.isclass(typing_) and issubclass(typing_, Measure):
        ann = (key, float)
        schema_element = schema_from_annotation(annotation=ann, schema_element=schema_element,
                                                order=order, editable=editable, title=title)
        schema_element[key]['units'] = typing_.units
    elif inspect.isclass(typing_) and issubclass(typing_, (BinaryFile, StringFile)):
        schema_element[key].update({'type': 'text', 'is_file': True})
    else:
        classname = dc.full_classname(object_=typing_, compute_for='class')
        if inspect.isclass(typing_) and issubclass(typing_, dc.DessiaObject):
            # Dessia custom classes
            schema_element[key].update({'type': 'object', 'standalone_in_db': typing_._standalone_in_db})
        # else:
            # DEPRECATED : Statically created dict structure
            # schema_element[key].update(static_dict_schema(typing_))
        schema_element[key]['classes'] = [classname]
    return schema_element


def schema_chunk(annotation, title: str = "", editable: bool = True, description: str = ""):
    if isinstance(annotation, str):
        raise ValueError

    if annotation in dc_types.TYPING_EQUIVALENCES:
        # Python Built-in type
        chunk = builtin_schema(annotation)
    elif dc_types.is_typing(annotation):
        chunk = typing_schema(typing_=annotation, title=title, editable=editable, description=description)
    elif hasattr(annotation, '__origin__') and annotation.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        chunk = {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
    elif annotation is Any:
        chunk = {'type': 'object', 'properties': {'.*': '.*'}}
    elif inspect.isclass(annotation) and issubclass(annotation, Measure):
        chunk = builtin_schema(float)
        chunk['units'] = annotation.si_unit
    elif inspect.isclass(annotation) and issubclass(annotation, (BinaryFile, StringFile)):
        chunk = {'type': 'text', 'is_file': True}
    elif inspect.isclass(annotation) and issubclass(annotation, dc.DessiaObject):
        # Dessia custom classes
        # class_schema = ClassSchema(annotation)
        # chunk = class_schema.write()
        classname = dc.full_classname(object_=annotation, compute_for='class')
        chunk = {'type': 'object', 'standalone_in_db': annotation._standalone_in_db, "classes": [classname]}
    else:
        raise NotImplementedError
    chunk.update({'title': title, 'editable': editable, 'description': description,
                  'python_typing': dc_types.serialize_typing(annotation)})
    return chunk


def typing_schema(typing_, title: str, editable: bool, description: str):
    origin = get_origin(typing_)
    if origin is Union:
        if dc_types.union_is_default_value(typing_):
            # This is a false Union => Is a default value set to None
            return schema_chunk(annotation=typing_, title=title, editable=editable, description=description)
        # Types union
        return schema_union_types(typing_)
    if origin in [list, collections.abc.Iterator]:
        # Homogenous sequences
        return schema_sequence_recursion(value=typing_, title=title, editable=editable)
    if origin is tuple:
        # Heterogenous sequences (tuples)
        return tuple_schema(typing_)
    if origin is dict:
        # Dynamically created dict structure)
        return dynamic_dict_schema(typing_)
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
        # schema_element[key].update({'type': 'object', 'instance_of': classname,
        #                                 'standalone_in_db': class_._standalone_in_db})
    if origin is dc_types.InstanceOf:
        # Several possible classes that are subclass of another one
        return instance_of_schema(typing_)
    if origin is MethodType or origin is ClassMethodType:
        return method_type_schema(annotation=typing_, editable=editable)
    if origin is type:
        return class_schema()
    raise NotImplementedError(f"Schema computation of typing {typing_} is not implemented")


def schema_sequence_recursion(value, title: str = None, editable: bool = False):
    if title is None:
        title = 'Items'
    chunk = {'type': 'array', 'python_typing': dc_types.serialize_typing(value)}

    items_type = get_args(value)[0]
    if dc_types.is_typing(items_type) and get_origin(items_type) is list:
        chunk['items'] = schema_sequence_recursion(value=items_type, title=title, editable=editable)
    else:
        chunk["items"] = schema_chunk(annotation=items_type, title=title, editable=editable, description="")
    return chunk


def set_default_value(schema_element, default_value):
    datatype = datatype_from_schema(schema_element)
    if default_value is None or datatype in ['builtin', 'heterogeneous_sequence', 'static_dict', 'dynamic_dict']:
        schema_element['default_value'] = default_value
    # elif datatype == 'builtin':
    #     schema_element[key]['default_value'] = default_value
    # elif datatype == 'heterogeneous_sequence':
    #     schema_element[key]['default_value'] = default_value
    elif datatype == 'homogeneous_sequence':
        msg = 'Object {} of type {} is not supported as default value'
        type_ = type(default_value)
        raise NotImplementedError(msg.format(default_value, type_))
    elif datatype in ['standalone_object', 'embedded_object', 'instance_of', 'union']:
        object_dict = default_value.to_dict()
        schema_element['default_value'] = object_dict
    return schema_element
    # if isinstance(default_value, tuple(TYPING_EQUIVALENCES.keys())) \
    #         or default_value is None:
    #     schema_element[key]['default_value'] = default_value
    # elif is_sequence(default_value):
    #     if datatype == 'heterogeneous_sequence':
    #         schema_element[key]['default_value'] = default_value
    #     else:
    #         msg = 'Object {} of type {} is not supported as default value'
    #         type_ = type(default_value)
    #         raise NotImplementedError(msg.format(default_value, type_))
    # else:
    #     if datatype in ['standalone_object', 'embedded_object',
    #                     'subclass', 'union']:
    #     object_dict = default_value.to_dict()
    #     schema_element[key]['default_value'] = object_dict
    #     else:


def builtin_schema(annotation):
    return {"type": dc_types.TYPING_EQUIVALENCES[annotation], 'python_typing': dc_types.serialize_typing(annotation)}


def tuple_schema(annotation):
    args = get_args(annotation)
    items = []
    for type_ in args:
        items.append({'type': dc_types.TYPING_EQUIVALENCES[type_]})
    return {'additionalItems': False, 'type': 'array', 'items': items}


def dynamic_dict_schema(annotation):
    args = get_args(annotation)
    key_type, value_type = args
    if key_type != str:
        # !!! Should we support other types ? Numeric ?
        raise NotImplementedError('Non strings keys not supported')
    if value_type not in dc_types.TYPING_EQUIVALENCES:
        raise ValueError(f'Dicts should have only builtins keys and values, got {value_type}')
    schema = {
        'type': 'object',
        'patternProperties': {
            '.*': {
                'type': dc_types.TYPING_EQUIVALENCES[value_type]
            }
        }
    }
    return schema


def instance_of_schema(annotation):
    args = get_args(annotation)
    class_ = args[0]
    classname = dc.full_classname(object_=class_, compute_for='class')
    return {'type': 'object', 'instance_of': classname, 'standalone_in_db': class_._standalone_in_db}


def method_type_schema(annotation, editable):
    origin = get_origin(annotation)
    class_type = get_args(annotation)[0]
    classmethod_ = origin is ClassMethodType
    chunk = schema_chunk(annotation=class_type, title="Class", editable=editable, description="")
    schema = {
        'type': 'object', 'is_method': True, 'classmethod_': classmethod_,
        'properties': {
            'class_': chunk,
            'name': {
                'type': 'string'
            }
        }
    }
    return schema


def class_schema():
    return {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
