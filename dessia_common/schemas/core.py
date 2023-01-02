#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema generation functions
"""
from copy import deepcopy
import inspect
import collections
import collections.abc
import typing as tp
import dessia_common.utils.types as dc_types
from dessia_common.abstract import CoreDessiaObject
from dessia_common.files import BinaryFile, StringFile
from dessia_common.typings import Subclass, MethodType, ClassMethodType, Any, InstanceOf
from dessia_common.measures import Measure
from dessia_common.utils.docstrings import parse_docstring, FAILED_DOCSTRING_PARSING, FAILED_ATTRIBUTE_PARSING
from dessia_common.utils.helpers import prettyname
import dessia_common.schemas.interfaces as si

SCHEMA_HEADER = {"definitions": {}, "$schema": "http://json-schema.org/d_raft-07/schema#",
                 "type": "object", "required": [], "properties": {}}
RESERVED_ARGNAMES = ['self', 'cls', 'progress_callback', 'return']

Issue = tp.TypedDict("Issue", {"attribute": str, "severity": str, "message": str})

# DessiaObjectType = tp.TypeVar("DessiaObjectType", bound=CoreDessiaObject)


class Schema:
    """
    Abstraction of a Schema.

    It reads the user-defined type hints and then writes into a dict the recursive structure of an object
    that can be handled by dessia_common.
    This dictionnary can then be translated as a json to be read by the frontend in order to compute edit forms,
    for example.
    """

    def __init__(self, annotations: si.Annotation, argspec, docstring: str):
        self.annotations = annotations
        self.attributes = [a for a in argspec.args if a not in RESERVED_ARGNAMES]

        self.property_schemas = {a: get_schema(annotations[a]) for a in self.attributes}

        # self.standalone_in_db = None
        # self.python_typing = ""

        # Parse docstring
        try:
            self.parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        except Exception:
            self.parsed_docstring = FAILED_DOCSTRING_PARSING

        self.parsed_attributes = self.parsed_docstring['attributes']

        self.required_arguments, self.default_arguments = split_default_args(argspec=argspec, merge=False)

    @property
    def editable_attributes(self):
        return [a for a in self.attributes if a not in RESERVED_ARGNAMES]

    def annotations_are_valid(self) -> tp.Tuple[bool, tp.List[Issue]]:
        """ Return wether the class definition is valid or not. """
        issues = []
        for attribute in self.attributes:
            schema = self.property_schemas[attribute]
            issues.extend(schema.check(attribute))
        return not any(issues), issues

    def chunk(self, attribute: str):
        """ Extract and compute a schema from one of the attributes. """
        schema = self.property_schemas[attribute]

        if self.parsed_attributes is not None and attribute in self.parsed_attributes:
            try:
                description = self.parsed_attributes[attribute]['desc']
            except Exception:
                description = FAILED_ATTRIBUTE_PARSING["desc"]
        else:
            description = ""

        editable = attribute in self.editable_attributes
        chunk = schema.write(title=prettyname(attribute), editable=editable, description=description)

        if attribute in self.default_arguments:
            # TODO Could use this and Optional proxy in order to inject real default values for mutables
            chunk = set_default_value(schema_element=chunk, default_value=self.default_arguments[attribute])
        return chunk

    @property
    def chunks(self):
        """ Concatenate schema chunks into a list. """
        return [self.chunk(a) for a in self.attributes]

    def write(self):
        """ Write the whole schema. """
        schema = deepcopy(SCHEMA_HEADER)
        properties = {a: self.chunk(a) for a in self.attributes}
        schema.update({"required": self.required_arguments, "properties": properties,
                       "description": self.parsed_docstring["description"]})
        return schema


class ClassSchema(Schema):
    """
    Schema of a class.

    Class must be a subclass of DessiaObject. It reads the __init__ annotations.
    """
    def __init__(self, class_: CoreDessiaObject):
        self.class_ = class_
        self.standalone_in_db = class_._standalone_in_db
        self.python_typing = str(class_)
        annotations = tp.get_type_hints(class_.__init__)

        members = inspect.getfullargspec(self.class_.__init__)
        docstring = class_.__doc__

        Schema.__init__(self, annotations=annotations, argspec=members, docstring=docstring)

    @property
    def editable_attributes(self):
        attributes = super().editable_attributes
        return [a for a in attributes if a not in self.class_._non_editable_attributes]

    def check(self) -> tp.Tuple[bool, tp.List[Issue]]:
        """ Check. """
        issues = []
        for attribute in self.attributes:
            if attribute not in self.annotations:
                msg = f"Property {attribute} has no typing"
                issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return not any(issues), issues


class MethodSchema(Schema):
    """
    Schema of a method.

    Given method should be one of a DessiaObject. It reads its annotations.
    """
    def __init__(self, method: tp.Callable):
        self.method = method

        annotations = tp.get_type_hints(method)
        Schema.__init__(self, annotations=annotations)


class Property:
    """ Base class for a schema property. """
    def __init__(self, annotation: tp.Type):
        self.annotation = annotation

    @property
    def schema(self):
        return self

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        return {'title': title, 'editable': editable, 'description': description,
                'python_typing': dc_types.serialize_typing(self.annotation), "type": None}

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Property Type Hint.

        Checks performed : None. TODO ?
        """
        return []


class TypingProperty(Property):
    """ Schema class for typing based annotations. """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    @property
    def args(self):
        return tp.get_args(self.annotation)

    @property
    def origin(self):
        return tp.get_origin(self.annotation)

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of TypingProperty Type Hint.

        Checks performed : None. TODO ?
        """
        return []


class Optional(TypingProperty):
    """
    Proxy Schema class for Optional properties.

    Optional is only a catch for arguments that default to None.
    Arguments with default values other than None are not considered Optionals
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    @property
    def schema(self):
        return get_schema(self.args[0])

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        default_value = None
        chunk = self.schema.write(title=title, editable=editable, description=description)
        chunk["default_value"] = default_value
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Optional proxy Type Hint.

        Checks performed : None TODO ?
        """
        return []


class Annotated(TypingProperty):
    """
    Proxy Schema class for annotated type hints.

    Annotated annotations are type hints with more arguments passed, such as value ranges, or probably enums,
    precision,...

    This could enable quite effective type checking on frontend form.

    Only available with python >= 3.11
    """
    _not_implemented_msg = "Annotated type hints are not implemented yet. This needs python 3.11 at least. " \
                           "Dessia only supports python 3.9 at the moment."

    # TODO Whenever Dessia decides to upgrade to python 3.11
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)
        raise NotImplementedError(self._not_implemented_msg)

    @property
    def schema(self):
        return get_schema(self.args[0])

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        raise NotImplementedError(self._not_implemented_msg)

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of DynamicDict Type Hint.

        Checks performed : None. TODO : Arg validity
        """
        raise NotImplementedError(self._not_implemented_msg)


class Builtin(Property):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = Property.write(self)
        chunk["type"] = dc_types.TYPING_EQUIVALENCES[self.annotation]
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Builtin Type Hint.

        Always return no issues
        """
        return []


class MeasureProperty(Builtin):
    def __init__(self, annotation: tp.Type[Measure]):
        super().__init__(annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        chunk["si_unit"] = self.annotation.si_unit
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Measure Type Hint.

        Checks performed :
        - Cannot be other than float. TODO : Is it possible ?
        """
        return []


class File(Property):
    def __init__(self, annotation: tp.Type):
        Property.__init__(self, annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        chunk.update({'type': 'text', 'is_file': True})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of File Type Hint.

        Checks performed : None. TODO ?
        """
        return []


class CustomClassProperty(Property):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)
        self.classname = dc_types.full_classname(object_=self.annotation, compute_for='class')

    @property
    def schema(self):
        return ClassSchema(self.annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'standalone_in_db': self.annotation._standalone_in_db,
                      "classes": [self.classname]})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of user custom class Type Hint.

        Checks performed :
        - Is subclass of DessiaObject TODO
        """
        issues = super().check(attribute)
        if not issubclass(self.annotation, CoreDessiaObject):
            issue = {"attribute": attribute, "severity": "error",
                     "message": f"Class '{self.classname}' is not a subclass of DessiaObject"}
            issues.append(issue)
        return issues


class Union(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        standalone_args = [a._standalone_in_db for a in self.args]
        if all(standalone_args):
            self.standalone = True
        elif not any(standalone_args):
            self.standalone = False
        else:
            self.standalone = None

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        classnames = [dc_types.full_classname(object_=a, compute_for='class') for a in self.args]
        chunk.update({'type': 'object', 'classes': classnames, 'standalone_in_db': self.standalone})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Union Type Hint.

        Checks performed :
        - Subobject are all standalone or none of them are. TODO : What happen if args are not DessiaObjects ?
        """
        issues = []
        standalone_args = [a._standalone_in_db for a in self.args]
        if not all(standalone_args) and any(standalone_args):
            issue = {"attribute": attribute, "severity": "error",
                     "message": f"standalone_in_db values for type '{self.annotation}' are not consistent"}
            issues.append(issue)
        return issues


class HeterogeneousSequence(TypingProperty):
    """ Datatype that can be seen as a tuple. Have any amount of arguments but a limited length. """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        self.items_schemas = [get_schema(a) for a in self.args]

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write()
        items = [sp.write() for sp in self.items_schemas]
        chunk.update({'type': 'array', 'additionalItems': False, 'items': items})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Tuple Type Hint.

        Checks performed :
        - Annotation has at least one argument, one for each element of the tuple
        """
        issues = super().check(attribute)
        if len(self.args) == 0:
            msg = f"Attribute '{attribute}' is typed as a 'Tuple' which requires at least 1 argument." \
                  f"Expected Tuple[Type0, Type1, ..., TypeN], got {self.annotation}."
            issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return issues


class HomogeneousSequence(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        self.items_schemas = [get_schema(a) for a in self.args]

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        if not title:
            title = 'Items'
        chunk = super().write(title=title, editable=editable, description=description)
        items_schemas = [sp.write(title=title, editable=editable, description=description) for sp in self.items_schemas]
        chunk.update({'type': 'array', 'python_typing': dc_types.serialize_typing(self.annotation),
                      "items": items_schemas})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of List Type Hint.

        Checks performed :
        - Annotation has exactly one argument, which is the type of all the element of the list.
        """
        issues = super().check(attribute)
        if len(self.args) != 1:
            msg = f"Attribute '{attribute}' is typed as a 'List' which requires exactly 1 argument." \
                  f"Expected List[Type], got {self.annotation}."
            issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return issues


class DynamicDict(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        key_type, value_type = self.args
        if key_type != str:
            # !!! Should we support other types ? Numeric ?
            raise NotImplementedError('Non strings keys not supported')
        if value_type not in dc_types.TYPING_EQUIVALENCES:
            raise ValueError(f'Dicts should have only builtins keys and values, got {value_type}')
        chunk = super().write(title=title, editable=editable, description=description)
        chunk.update({'type': 'object',
                      'patternProperties': {
                          '.*': {
                            'type': dc_types.TYPING_EQUIVALENCES[value_type]
                          }
                      }})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of DynamicDict Type Hint.

        Checks performed :
        - Annotation as exactly two arguments, first one for keys, second one for values
        - Key Type is str TODO
        - Value Type is simple TODO
        """
        issues = super().check(attribute)
        if len(self.args) != 2:
            msg = f"Attribute '{attribute}' is typed as a 'Dict' which requires exactly 2 arguments." \
                  f"Expected Dict[KeyType, ValueType], got {self.annotation}."
            issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return issues


class InstanceOfProperty(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        class_ = self.args[0]
        classname = dc_types.full_classname(object_=class_, compute_for='class')
        chunk.update({'type': 'object', 'instance_of': classname, 'standalone_in_db': class_._standalone_in_db})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of InstanceOf Type Hint.

        Checks performed :
        - Annotation has exactly one argument, which is the type of the base class.
        """
        issues = super().check(attribute)
        if len(self.args) != 1:
            msg = f"Attribute '{attribute}' is typed as a 'InstanceOf' which requires exactly 1 argument." \
                  f"Expected 'InstanceOf[Type]', got '{self.annotation}'."
            issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return issues


class SubclassProperty(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        raise NotImplementedError("Subclass is not implemented yet")

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Subclass Type Hint.

        Checks performed :
        - Annotation has exactly one argument, which is the type of the base class.
        """
        issues = super().check(attribute)
        if len(self.args) != 1:
            msg = f"Attribute '{attribute}' is typed as a 'Subclass' which requires exactly 1 argument." \
                  f"Expected 'Subclass[Type]', got '{self.annotation}'."
            issues.append({"attribute": attribute, "severity": "error", "message": msg})
        return issues


class MethodTypeProperty(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        self.class_ = self.args[0]
        self.class_schema = get_schema(self.class_)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        classmethod_ = self.origin is ClassMethodType
        chunk.update({
            'type': 'object', 'is_method': True, 'classmethod_': classmethod_,
            'properties': {
                'class_': self.class_schema,
                'name': {
                    'type': 'string'
                }
            }
        })
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of MethodType Type Hint.

        Checks performed :
        - Class has method TODO
        """
        return []


class ClassProperty(TypingProperty):
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def write(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().write(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}})
        return chunk

    def check(self, attribute: str) -> tp.List[Issue]:
        """
        Check validity of Class Type Hint.

        Checks performed :
        - Annotation has no argument TODO
        """
        return []


def inspect_arguments(method: tp.Callable, merge: bool = False):
    """
    Find default value and required arguments of class construction.

    Get method arguments and default arguments as sequences while removing forbidden ones (self, cls...).
    """
    argspec = inspect.getfullargspec(method)
    return split_default_args(argspec=argspec, merge=merge)


def split_default_args(argspec: inspect.FullArgSpec, merge: bool = False):
    nargs, ndefault_args = split_argspecs(argspec)

    default_arguments = {}
    arguments = []
    for iargument, argument in enumerate(argspec.args[1:]):
        if argument not in RESERVED_ARGNAMES:
            if iargument >= nargs - ndefault_args:
                default_value = argspec.defaults[ndefault_args - nargs + iargument]
                if merge:
                    arguments.append((argument, default_value))
                else:
                    default_arguments[argument] = default_value
            else:
                arguments.append(argument)
    return arguments, default_arguments


def split_argspecs(argspecs: inspect.FullArgSpec) -> tp.Tuple[int, int]:
    """ Get number of regular arguments as well as arguments with default values. """
    nargs = len(argspecs.args) - 1

    if argspecs.defaults is not None:
        ndefault_args = len(argspecs.defaults)
    else:
        ndefault_args = 0
    return nargs, ndefault_args


def get_schema(annotation: tp.Type) -> Property:
    print(annotation)
    if annotation in dc_types.TYPING_EQUIVALENCES:
        return Builtin(annotation)
    if dc_types.is_typing(annotation):
        return get_typing_schema(annotation)
    if hasattr(annotation, '__origin__') and annotation.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        # return {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
        pass
    if annotation is Any:
        # chunk = {'type': 'object', 'properties': {'.*': '.*'}}
        pass
    if inspect.isclass(annotation):
        return custom_class_schema(annotation)
    raise NotImplementedError(f"No schema defined for annotation '{annotation}'.")


def get_typing_schema(typing_) -> Property:
    origin = tp.get_origin(typing_)
    print(origin)
    if origin is tp.Union:
        if dc_types.union_is_default_value(typing_):
            # This is a false Union => Is a default value set to None
            return Optional(typing_)
        # Types union
        return Union(typing_)
    if origin is tuple:
        return HeterogeneousSequence(typing_)
    if origin in [list, collections.abc.Iterator]:
        return HomogeneousSequence(typing_)
    if origin is dict:
        return DynamicDict(typing_)
    if origin is Subclass:
        pass
    if origin is InstanceOf:
        return InstanceOfProperty(typing_)
    if origin in [MethodType, ClassMethodType]:
        return MethodTypeProperty(typing_)
    if origin is type:
        return ClassProperty(typing_)
    raise NotImplementedError(f"No Schema defined for typing '{typing_}'.")


def custom_class_schema(annotation: tp.Type) -> Property:
    if issubclass(annotation, Measure):
        return MeasureProperty(annotation)
    if issubclass(annotation, (BinaryFile, StringFile)):
        return File(annotation)
    if issubclass(annotation, CoreDessiaObject):
        # Dessia custom classes
        return CustomClassProperty(annotation)
    raise NotImplementedError(f"No Schema defined for type '{annotation}'.")


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
    args = tp.get_args(annotation)
    classnames = [dc_types.full_classname(object_=a, compute_for='class') for a in args]
    standalone_args = [a._standalone_in_db for a in args]
    if all(standalone_args):
        standalone = True
    elif not any(standalone_args):
        standalone = False
    else:
        raise ValueError(f"standalone_in_db values for type '{annotation}' are not consistent")
    return {'type': 'object', 'classes': classnames, 'standalone_in_db': standalone}


def schema_from_annotation(annotation, schema_element, order, editable=None, title=None, parsed_attributes=None):
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
    schema_element[key] = {'title': title, 'editable': editable, 'order': order, 'description': description,
                           'python_typing': dc_types.serialize_typing(typing_)}

    if typing_ in dc_types.TYPING_EQUIVALENCES:
        # Python Built-in type
        schema_element[key]['type'] = dc_types.TYPING_EQUIVALENCES[typing_]

    elif dc_types.is_typing(typing_):
        origin = tp.get_origin(typing_)
        args = tp.get_args(typing_)
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
        schema_element[key]['si_unit'] = typing_.si_unit
    elif inspect.isclass(typing_) and issubclass(typing_, (BinaryFile, StringFile)):
        schema_element[key].update({'type': 'text', 'is_file': True})
    else:
        classname = dc_types.full_classname(object_=typing_, compute_for='class')
        if inspect.isclass(typing_) and issubclass(typing_, CoreDessiaObject):
            # Dessia custom classes
            schema_element[key].update({'type': 'object', 'standalone_in_db': typing_._standalone_in_db})
        # else:
            # DEPRECATED : Statically created dict structure
            # schema_element[key].update(static_dict_schema(typing_))
        schema_element[key]['classes'] = [classname]
    return schema_element


def schema_chunk(annotation, title: str, editable: bool, description: str):
    if isinstance(annotation, str):
        raise ValueError

    if annotation in dc_types.TYPING_EQUIVALENCES:  # TODO DONE
        # Python Built-in type
        chunk = builtin_schema(annotation)
    elif dc_types.is_typing(annotation):  # TODO DONE
        chunk = typing_schema(typing_=annotation, title=title, editable=editable, description=description)
    elif hasattr(annotation, '__origin__') and annotation.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        chunk = {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
    elif annotation is Any:
        chunk = {'type': 'object', 'properties': {'.*': '.*'}}
    elif inspect.isclass(annotation) and issubclass(annotation, Measure):  # TODO DONE
        chunk = schema_chunk(annotation=float, title=title, editable=editable, description=description)
        chunk['si_unit'] = annotation.si_unit
    elif inspect.isclass(annotation) and issubclass(annotation, (BinaryFile, StringFile)):  # TODO DONE
        chunk = {'type': 'text', 'is_file': True}
    elif inspect.isclass(annotation) and issubclass(annotation, CoreDessiaObject):  # TODO DONE
        # Dessia custom classes
        classname = dc_types.full_classname(object_=annotation, compute_for='class')
        chunk = {'type': 'object', 'standalone_in_db': annotation._standalone_in_db, "classes": [classname]}
    else:
        raise NotImplementedError(f"Annotation {annotation} is not supported.")
    chunk.update({'title': title, 'editable': editable, 'description': description,
                  'python_typing': dc_types.serialize_typing(annotation)})
    return chunk


def typing_schema(typing_, title: str, editable: bool, description: str):
    origin = tp.get_origin(typing_)
    if origin is Union:  # TODO DONE
        if dc_types.union_is_default_value(typing_):  # TODO DONE
            # This is a false Union => Is a default value set to None
            return schema_chunk(annotation=typing_, title=title, editable=editable, description=description)
        # Types union  # TODO DONE
        return schema_union_types(typing_)
    if origin in [list, collections.abc.Iterator]:  # TODO DONE
        # Homogenous sequences
        return schema_sequence_recursion(value=typing_, title=title, editable=editable)
    if origin is tuple:  # TODO DONE
        # Heterogenous sequences (tuples)
        return tuple_schema(typing_)
    if origin is dict:  # TODO DONE
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
    if origin is dc_types.InstanceOf:  # TODO DONE
        # Several possible classes that are subclass of another one
        return instance_of_schema(typing_)
    if origin is MethodType or origin is ClassMethodType:  # TODO DONE
        return method_type_schema(annotation=typing_, editable=editable)
    if origin is type:  # TODO DONE
        return class_schema()
    raise NotImplementedError(f"Schema computation of typing {typing_} is not implemented")


def schema_sequence_recursion(value, title: str = None, editable: bool = False):
    if title is None:
        title = 'Items'
    chunk = {'type': 'array', 'python_typing': dc_types.serialize_typing(value)}

    items_type = tp.get_args(value)[0]
    if dc_types.is_typing(items_type) and tp.get_origin(items_type) is list:
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


def builtin_schema(annotation):  # TODO DONE
    chunk = {"type": dc_types.TYPING_EQUIVALENCES[annotation], 'python_typing': dc_types.serialize_typing(annotation)}
    return chunk


def tuple_schema(annotation):  # TODO DONE
    args = tp.get_args(annotation)
    items = []
    for type_ in args:
        # TODO Should classes other than builtins be allowed here ?
        items.append({'type': dc_types.TYPING_EQUIVALENCES[type_]})
    return {'additionalItems': False, 'type': 'array', 'items': items}


def dynamic_dict_schema(annotation):  # TODO DONE
    args = tp.get_args(annotation)
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


def instance_of_schema(annotation):  # TODO DONE
    args = tp.get_args(annotation)
    class_ = args[0]
    classname = dc_types.full_classname(object_=class_, compute_for='class')
    return {'type': 'object', 'instance_of': classname, 'standalone_in_db': class_._standalone_in_db}


def method_type_schema(annotation, editable):  # TODO DONE
    origin = tp.get_origin(annotation)
    class_type = tp.get_args(annotation)[0]
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


def class_schema():  # TODO DONE
    return {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
