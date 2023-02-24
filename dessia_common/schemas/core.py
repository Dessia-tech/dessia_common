#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Schema generation functions. """
import re
import warnings
from copy import deepcopy
import inspect
import collections.abc
from typing import Tuple, Dict, List, Type, get_args, get_origin, get_type_hints, Callable, Union,\
    TypeVar, TypedDict, Optional, Any
from functools import cached_property
from dessia_common.utils.helpers import full_classname, get_python_class_from_class_name
from dessia_common.abstract import CoreDessiaObject
from dessia_common.files import BinaryFile, StringFile
from dessia_common.typings import MethodType, ClassMethodType, InstanceOf, Subclass
from dessia_common.measures import Measure
from dessia_common.utils.helpers import prettyname
from dessia_common.schemas.interfaces import Annotations, T
from dessia_common.checks import CheckList, FailedCheck, PassedCheck, CheckWarning

SCHEMA_HEADER = {"definitions": {}, "$schema": "http://json-schema.org/draft-07/schema#",
                 "type": "object", "required": [], "properties": {}}
RESERVED_ARGNAMES = ['self', 'cls', 'progress_callback', 'return']
TYPING_EQUIVALENCES = {int: 'number', float: 'number', bool: 'boolean', str: 'string'}
TYPES_FROM_STRING = {'unicode': str, 'str': str, 'float': float, 'int': int, 'bool': bool}

_fullargsspec_cache = {}


class UntypedArgument(FailedCheck):
    """ Used when an argument is not typed in function or class definition. """

    def __init__(self, attribute: str):
        super().__init__(f"Attribute '{attribute}' : has no typing")


class WrongNumberOfArguments(FailedCheck):
    """ Used when a typing does not have the right amount of arguments. """


class WrongType(FailedCheck):
    """ Used when an annotation does not have the right type. """


class UnsupportedDefault(FailedCheck):
    """ Used when an argument defines a default that is not supported. """


class Schema:
    """
    Abstraction of a Schema.

    It reads the user-defined type hints and then writes into a Dict the recursive structure of an object
    that can be handled by dessia_common.
    This dictionnary can then be translated as a json to be read by the frontend in order to compute edit forms,
    for example.

    Right now Schema doesn't inherit from any DessiaObject class (SerializableObject ?), but could, in the future.
    That is why it implements methods with the same name.

    TODO We might want to define our 'own argspecs', in order to full native support of workflow.
     We could translate inspect argspecs into a dessia_common pseudo-language.
    """

    def __init__(self, annotations: Annotations, argspec: inspect.FullArgSpec, docstring: str, name: str = ""):
        self.annotations = annotations
        self.attributes = [a for a in argspec.args if a not in RESERVED_ARGNAMES]
        self.name = name

        try:  # Parse docstring
            self.parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        except Exception:  # Catching broad exception because we don't want it to break platform if a failure occurs
            self.parsed_docstring = FAILED_DOCSTRING_PARSING

        self.parsed_attributes = self.parsed_docstring['attributes']

        self.required_arguments, self.default_arguments = split_default_args(argspecs=argspec, merge=False)

        self.property_schemas = {}
        for attribute in self.attributes:
            if attribute in self.annotations:
                default = self.default_arguments.get(attribute, None)
                annotation = self.annotations[attribute]
                schema = get_schema(annotation=annotation, attribute=attribute, definition_default=default)
                self.property_schemas[attribute] = schema

    @property
    def editable_attributes(self):
        """ Attributes that are not in RESERVED_ARGNAMES. """
        return [a for a in self.attributes if a not in RESERVED_ARGNAMES]

    def chunk(self, attribute: str):
        """ Extract and compute a schema from one of the attributes. """
        schema = self.property_schemas.get(attribute, None)

        if self.parsed_attributes is not None and attribute in self.parsed_attributes:
            try:
                description = self.parsed_attributes[attribute]['desc']
            except Exception:  # Catching broad exception because we don't want it to break platform if a failure occurs
                description = FAILED_ATTRIBUTE_PARSING["desc"]
        else:
            description = ""

        editable = attribute in self.editable_attributes
        if schema is not None:
            return schema.to_dict(title=prettyname(attribute), editable=editable, description=description)

        # if attribute in self.default_arguments:
        #     # TODO Could use this and Optional proxy in order to inject real default values for mutable
        #     default = self.default_arguments.get(attribute, None)
        #     print("Default", default)
        #     chunk["default_value"] = schema.default_value(definition_default=default)
        return {}

    @property
    def chunks(self):
        """ Concatenate schema chunks into a List. """
        return [self.chunk(a) for a in self.attributes]

    def to_dict(self):
        """ Write the whole schema. """
        schema = deepcopy(SCHEMA_HEADER)
        properties = {a: self.chunk(a) for a in self.attributes}
        schema.update({"required": self.required_arguments, "properties": properties,
                       "description": self.parsed_docstring["description"]})
        return schema

    def default_dict(self):
        """
        Compute global default Dict.

        If a definition default have been set by user, most schemas will return this value (or serialized).
        if not, schemas will compute a default compatible with platform (None most of the time).
        """
        return {a: self.property_schemas[a].default_value() for a in self.attributes}

    def check_list(self) -> CheckList:
        """
        Browse all properties and List potential issues.

        Checks performed for each argument :
        - Is typed in method definition
        - Schema specific check
        """
        issues = CheckList([])

        for attribute in self.attributes:
            # Is typed
            is_typed_check = self.attribute_is_annotated(attribute)
            issues += CheckList([is_typed_check])

            if is_typed_check.level != "error":
                # Specific check
                schema = self.property_schemas[attribute]
                issues += schema.check_list()
        return issues

    @property
    def is_valid(self) -> bool:
        """ Return whether the class definition is valid or not. """
        return not self.check_list().checks_above_level("error")

    def attribute_is_annotated(self, attribute: str) -> PassedCheck:
        """ Check whether given attribute is annotated in function definition or not. """
        if attribute not in self.annotations:
            return UntypedArgument(attribute)
        return PassedCheck(f"Attribute '{attribute}' : is annotated")


class ClassSchema(Schema):
    """
    Schema of a class.

    Class must be a subclass of DessiaObject. It reads the __init__ annotations.
    """

    def __init__(self, class_: Type[CoreDessiaObject]):
        self.class_ = class_
        self.standalone_in_db = class_._standalone_in_db
        self.python_typing = full_classname(class_, compute_for="class")
        annotations = get_type_hints(class_.__init__)

        members = inspect.getfullargspec(self.class_.__init__)
        docstring = class_.__doc__

        Schema.__init__(self, annotations=annotations, argspec=members, docstring=docstring,
                        name=full_classname(class_))

    @property
    def editable_attributes(self):
        """ Attributes that are not in RESERVED_ARGNAMES nor defined as non editable by user. """
        attributes = super().editable_attributes
        return [a for a in attributes if a not in self.class_._non_editable_attributes]

    def default_dict(self):
        """ Compute class default Dict. Add object_class to base one. """
        dict_ = super().default_dict()
        dict_["object_class"] = self.python_typing
        return dict_


class MethodSchema(Schema):
    """
    Schema of a method.

    Given method should be one of a DessiaObject. It reads its annotations.
    """

    def __init__(self, method: Callable):
        if isinstance(method, property):
            method = method.fget
        self.method = method

        annotations = get_type_hints(method)
        members = inspect.getfullargspec(method)
        self.return_annotation = annotations.get("return", None)
        docstring = method.__doc__
        super().__init__(annotations=annotations, argspec=members, docstring=docstring, name=method.__name__)

        self.required_arguments = [str(self.attributes.index(a)) for a in self.required_arguments]

    @property
    def serialized(self):
        return {k: s.serialized for k, s in self.property_schemas.items()}

    @property
    def return_schema(self):
        return get_schema(annotation=self.return_annotation, attribute="return")

    @property
    def return_serialized(self):
        try:
            return self.return_schema.serialized
        except NotImplementedError:
            return None

    def to_dict(self):
        """ Write the whole schema. """
        schema = deepcopy(SCHEMA_HEADER)
        properties = {str(i): self.chunk(a) for i, a in enumerate(self.attributes)}
        schema.update({"required": self.required_arguments, "properties": properties,
                       "description": self.parsed_docstring["description"]})
        return schema

    def definition_json(self):
        return {"name": self.name, "return": self.return_serialized, "arguments": self.serialized,
                "valid": self.is_valid, "checks": self.check_list().to_dict()["checks"]}

    def check_list(self) -> CheckList:
        """
        Browse all properties and List potential issues.

        Checks performed for each argument :
        - Is typed in method definition
        - Schema specific check
        """
        issues = super().check_list()
        issues += CheckList([self.return_is_annotated(), self.return_type_is_valid()])
        return issues

    def return_is_annotated(self) -> PassedCheck:
        """ Check whether method return is annotated in definition or not. """
        if "return" not in self.annotations:
            return CheckWarning(f"Method return : is not annotated")
        return PassedCheck(f"Method return : is annotated")

    def return_type_is_valid(self) -> PassedCheck:
        """ Check whether given attribute is annotated in function definition or not. """
        try:
            _ = self.return_schema
            return PassedCheck(f"Method return : '{self.return_annotation}' is valid")
        except NotImplementedError:
            return CheckWarning(f"Method return : '{self.return_annotation} is not valid")


class Property:
    """ Base class for a schema property. """

    def __init__(self, annotation: Type[T], attribute: str, definition_default: T = None):
        self.annotation = annotation
        self.attribute = attribute
        self.definition_default = definition_default

    @property
    def schema(self):
        """ Return a reference to itself. Might be overwritten for proxy such as Optional or Annotated. """
        return self

    @cached_property
    def serialized(self) -> str:
        """ Stringified annotation. """
        return str(self.annotation)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Simple get class from its name. """
        return get_python_class_from_class_name(serialized)

    @property
    def check_prefix(self) -> str:
        """ Shortcut for Check message prefixes. """
        return f"Attribute '{self.attribute}' : "

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write base schema as a Dict. """
        return {'title': title, 'editable': editable, 'description': description,
                'python_typing': self.serialized, "type": None}

    def default_value(self):
        """ Generic default. Yield user default if defined, else None. """
        return self.definition_default

    def check_list(self) -> CheckList:
        """
        Check validity of Property Type Hint.

        Checks performed : None. TODO ?
        """
        return CheckList([])


class TypingProperty(Property):
    """ Schema class for typing based annotations. """

    SERIALIZED_REGEXP = r"([^\[\]]*)\[(.*)\]"

    def __init__(self, annotation: Type[T], attribute: str, definition_default: T = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @property
    def args(self) -> Tuple[Type[T], ...]:
        """ Return Typing arguments. """
        return get_args(self.annotation)

    @property
    def origin(self) -> Type:
        """ Return Typing origin. """
        return get_origin(self.annotation)

    @property
    def args_schemas(self) -> List[Property]:
        """ Get schema for each argument. """
        return [get_schema(annotation=a, attribute=f"{self.attribute}/{i}") for i, a in enumerate(self.args)]

    @cached_property
    def serialized(self) -> str:
        """ Recursively stringify annotation. """
        serialized = self.origin.__name__
        if serialized in ["list", "dict", "tuple", "type"]:
            # TODO Dirty quickfix. Find a generic way to automatize this
            serialized = serialized.capitalize()
        if self.args:
            return compute_typing_schema_serialization(serialized_typing=serialized, args_schemas=self.args_schemas)
        return serialized

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Split Typing and Args and delegate deserialization to specific classes. """
        typename = cls.type_from_serialized(serialized)
        schema_class = SERIALIZED_TO_SCHEMA_CLASS[typename]
        return schema_class.annotation_from_serialized(serialized)

    @classmethod
    def type_from_serialized(cls, serialized: str) -> str:
        """ Get Typing from serialized value. """
        return re.match(cls.SERIALIZED_REGEXP, serialized).group(1)

    @classmethod
    def _raw_args_from_serialized(cls, serialized: str) -> str:
        """ Get args as str from serialized value. """
        if "[" in serialized and "]" in serialized:
            args = re.match(cls.SERIALIZED_REGEXP, serialized).group(2)
            return args.replace(" ", "")
        return ""

    @classmethod
    def _args_from_serialized(cls, serialized: str) -> Tuple[Type[T]]:
        """ Deserialize args. """
        rawargs = cls._raw_args_from_serialized(serialized)
        args = extract_args(rawargs)
        return tuple([deserialize_annotation(a) for a in args])

    @classmethod
    def unfold_serialized_annotation(cls, serialized: str):
        """ Get Typing and Args as strings. """
        return re.match(cls.SERIALIZED_REGEXP, serialized).groups()

    def has_one_arg(self) -> PassedCheck:
        """ Annotation should have exactly one argument. """
        if len(self.args) != 1:
            pretty_origin = prettyname(self.origin.__name__)
            msg = f"{self.check_prefix}is typed as a '{pretty_origin}' which requires exactly 1 argument. " \
                  f"Expected '{pretty_origin}[T]', got '{self.annotation}'."
            return WrongNumberOfArguments(msg)
        return PassedCheck(f"{self.check_prefix}has exactly one arg in its definition.")


class ProxyProperty(TypingProperty):
    """
    Schema Class for Proxies.

    Proxies are just intermediate types which actual schemas if its args. For example OptionalProperty proxy.
    """

    def __init__(self, annotation: Type[T], attribute: str, definition_default: T = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

        self.annotation = self.args[0]

    @property
    def schema(self):
        """ Return a reference to its only arg. """
        return get_schema(annotation=self.annotation, attribute=self.attribute,
                          definition_default=self.definition_default)

    @cached_property
    def serialized(self) -> str:
        """ Stringify under-proxy-annotation. """
        return self.schema.serialized

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Proxy annotation. """
        raise NotImplementedError(f"Cannot deserialize annotation '{serialized}' as Proxy.")


class OptionalProperty(ProxyProperty):
    """
    Proxy Schema class for OptionalProperty properties.

    OptionalProperty is only a catch for arguments that default to None.
    Arguments with default values other than None are not considered Optionals
    """

    def __init__(self, annotation: Type[T], attribute: str, definition_default: T = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Optional as a Dict. """
        default_value = self.schema.default_value()
        chunk = self.schema.to_dict(title=title, editable=editable, description=description)
        chunk["default_value"] = default_value
        return chunk

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Optional annotation. """
        raise NotImplementedError("Optional deser not implemented yet. ")


class AnnotatedProperty(ProxyProperty):
    """
    Proxy Schema class for annotated type hints.

    AnnotatedProperty annotations are type hints with more arguments passed, such as value ranges, or probably enums,
    precision,...

    This could enable quite effective type checking on frontend form.

    Only available with python >= 3.11
    """

    _not_implemented_msg = "AnnotatedProperty type hints are not implemented yet. This needs python 3.11 at least. " \
                           "Dessia only supports python 3.9 at the moment."

    # TODO Whenever Dessia decides to upgrade to python 3.11
    def __init__(self, annotation: Type[T], attribute: str, definition_default: T = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)
        raise NotImplementedError(self._not_implemented_msg)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Annotated annotation. """
        raise NotImplementedError(cls._not_implemented_msg)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Annotated as a Dict. """
        raise NotImplementedError(self._not_implemented_msg)

    def check_list(self) -> CheckList:
        """
        Check validity of DynamicDict Type Hint.

        Checks performed : None. TODO : Arg validity
        """
        raise NotImplementedError(self._not_implemented_msg)


Builtin = Union[str, bool, float, int]


class BuiltinProperty(Property):
    """ Schema class for Builtin type hints. """

    def __init__(self, annotation: Type[Builtin], attribute: str, definition_default: Builtin = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @cached_property
    def serialized(self) -> str:
        """ Builtin name. """
        return self.annotation.__name__

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Get real Type from types dictionnary. """
        return TYPES_FROM_STRING[serialized]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Builtin as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk["type"] = TYPING_EQUIVALENCES[self.annotation]
        if self.default_value() is not None:
            chunk["default_value"] = self.default_value()
        return chunk


class MeasureProperty(BuiltinProperty):
    """ Schema class for Measure type hints. """

    def __init__(self, annotation: Type[Measure], attribute: str, definition_default: Measure = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @cached_property
    def serialized(self) -> str:
        """ Full class name. """
        return full_classname(object_=self.annotation, compute_for="class")

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize as annotation custom class. """
        return Property.annotation_from_serialized(serialized)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Measure as a Dict. """
        chunk = Property.to_dict(self, title=title, editable=editable, description=description)
        chunk.update({"si_unit": self.annotation.si_unit, "type": "number"})
        return chunk


File = Union[StringFile, BinaryFile]


class FileProperty(Property):
    """ Schema class for File type hints. """

    def __init__(self, annotation: Type[File], attribute: str, definition_default: File = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @property
    def serialized(self) -> str:
        """ Return file classname. """
        return full_classname(object_=self.annotation, compute_for="class")

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write File as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'text', 'is_file': True})
        return chunk

    def check_list(self) -> CheckList:
        """
        Check validity of File Type Hint.

        Checks performed :
        - Doesn't define any default value.
        """
        issues = super().check_list()
        issues += CheckList([self.has_no_default()])
        return issues

    def has_no_default(self) -> PassedCheck:
        """ Check if the user definition doesn't have any default value, as it is not supported for files. """
        if self.definition_default is not None:
            msg = f"{self.check_prefix}File input defines a default value, whereas it is not supported."
            return CheckWarning(msg)
        msg = f"{self.check_prefix}File input doesn't define a default value, as it should."
        return PassedCheck(msg)


class CustomClass(Property):
    """ Schema class for CustomClass type hints. """

    def __init__(self, annotation: Type[CoreDessiaObject], attribute: str,
                 definition_default: CoreDessiaObject = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @property
    def schema(self):
        """ Return a reference to the schema of the annotation. """
        return ClassSchema(self.annotation)

    @cached_property
    def serialized(self) -> str:
        """ Full class name. """
        return full_classname(object_=self.annotation, compute_for='class')

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write CustomClass as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'standalone_in_db': self.annotation._standalone_in_db,
                      "classes": [self.serialized]})
        return chunk

    def default_value(self):
        """ Default value for an object. """
        return object_default(definition_default=self.definition_default, class_schema=self.schema)

    def check_list(self) -> CheckList:
        """
        Check validity of user custom class Type Hint.

        Checks performed :
        - Is subclass of DessiaObject
        """
        issues = super().check_list()
        issues += CheckList([self.is_dessia_object_typed()])
        return issues

    def is_dessia_object_typed(self) -> PassedCheck:
        """ Check whether if typing for given attribute annotates a subclass of DessiaObject or not . """
        if not issubclass(self.annotation, CoreDessiaObject):
            return WrongType(f"{self.check_prefix}Class '{self.serialized}' is not a subclass of DessiaObject.")
        msg = f"{self.check_prefix}Class '{self.serialized}' is properly typed as a subclass of DessiaObject."
        return PassedCheck(msg)


class UnionProperty(TypingProperty):
    """ Schema class for Union type hints. """

    def __init__(self, annotation: Type[Union[T]], attribute: str, definition_default: Union[T] = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

        standalone_args = [a._standalone_in_db for a in self.args]
        if all(standalone_args):
            self.standalone = True
        elif not any(standalone_args):
            self.standalone = False
        else:
            self.standalone = None

    @property
    def serialized(self) -> str:
        """ Generic serialization with 'Union' enforced, because Union annotation has no __name__ attribute. """
        return compute_typing_schema_serialization(serialized_typing="Union", args_schemas=self.args_schemas)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Union annotation. """
        return Union[TypingProperty._args_from_serialized(serialized)]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Union as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'classes': [self.serialized], 'standalone_in_db': self.standalone})
        return chunk

    def default_value(self):
        """ Default value for an object. """
        return object_default(self.definition_default)

    def check_list(self) -> CheckList:
        """
        Check validity of UnionProperty Type Hint.

        Checks performed :
        - Subobject are all standalone or none of them are. TODO : What happen if args are not DessiaObjects ?
        """
        issues = super().check_list()
        issues += CheckList([self.classes_are_standalone_consistent()])
        return issues

    def classes_are_standalone_consistent(self) -> PassedCheck:
        """ Check whether all class in Union are standalone or none of them are. """
        standalone_args = [a._standalone_in_db for a in self.args]
        if all(standalone_args):
            msg = f"{self.check_prefix}All arguments of Union type '{self.annotation}' are standalone in db."
            return PassedCheck(msg)
        if not any(standalone_args):
            msg = f"{self.check_prefix}No arguments of Union type '{self.annotation}' are standalone in db."
            return PassedCheck(msg)
        msg = f"{self.check_prefix}'standalone_in_db' values for arguments of Union type '{self.annotation}'" \
              f"are not consistent. They should be all standalone in db or none of them should."
        return WrongType(msg)


class HeterogeneousSequence(TypingProperty):
    """
    Schema class for Tuple type hints.

    Datatype that can be seen as a Tuple. Have any amount of arguments but a limited length.
    """

    def __init__(self, annotation: Type[Tuple], attribute: str, definition_default: Tuple = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

        self.additional_items = Ellipsis in self.args

    @property
    def args_schemas(self) -> List[Property]:
        """
        If length is undefined (additional_items is True), then we only have one possible argument type.

        Otherwise, each argument is ordered and strictly defined by its type
        """
        if self.additional_items:
            return [get_schema(annotation=self.args[0], attribute=f"{self.attribute}/0")]
        return [get_schema(annotation=a, attribute=f"{self.attribute}/{i}") for i, a in enumerate(self.args)]

    @property
    def serialized(self) -> str:
        """ If additional items, concatenate ellipsis. """
        if self.additional_items:
            return f"Tuple[{self.args_schemas[0].serialized}, ...]"
        return super().serialized

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Tuple annotation with undefined length support. """
        rawargs = cls._raw_args_from_serialized(serialized)
        if ",..." in rawargs:
            arg = rawargs.split(",...")[0]
            subtype = deserialize_annotation(arg)
            return Tuple[subtype, ...]
        return Tuple[TypingProperty._args_from_serialized(serialized)]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write HeterogeneousSequence as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        items = [sp.to_dict(title=f"{title}/{i}", editable=editable) for i, sp in enumerate(self.args_schemas)]
        chunk.update({'type': 'array', 'additionalItems': self.additional_items, 'items': items})
        return chunk

    def default_value(self):
        """
        Default value for a Tuple.

        Return serialized user default if defined, else a Tuple of Nones with the right size.
        """
        if self.definition_default is not None:
            return self.definition_default
        return tuple(s.default_value() for s in self.args_schemas)

    def check_list(self) -> CheckList:
        """ Check validity of Tuple Type Hint. """
        issues = super().check_list()
        issues += CheckList([self.has_enough_args(), self.ellipsis_has_exactly_two_args()])
        return issues

    def has_enough_args(self) -> PassedCheck:
        """ Annotation should have at least one argument, one for each element of the Tuple. """
        if len(self.args) == 0:
            msg = f"{self.check_prefix}is typed as a 'Tuple' which requires at least 1 argument. " \
                  f"Expected 'Tuple[T0, T1, ..., Tn]', got '{self.annotation}'."
            return WrongNumberOfArguments(msg)
        return PassedCheck(f"{self.check_prefix}has at least one argument : '{self.annotation}'.")

    def ellipsis_has_exactly_two_args(self) -> PassedCheck:
        """
        Tuple can be ellipsed (Tuple[T, ...]), meaning that it contains any number of element.

        In this case it MUST have exactly two arguments.
        """
        if self.additional_items and len(self.args) != 2:
            msg = f"{self.check_prefix}is typed as an ellipsed 'Tuple' which requires at exactaly 2 arguments. " \
                  f"Expected 'Tuple[T, ...]', got '{self.annotation}'."
            return WrongNumberOfArguments(msg)
        return PassedCheck(f"{self.check_prefix}is not an ill-defined ellipsed tuple : '{self.annotation}'.")


class HomogeneousSequence(TypingProperty):
    """
    Schema class for List type hints.

    Datatype that can be seen as a List. Have only one argument but an unlimited length.
    """

    def __init__(self, annotation: Type[List[T]], attribute: str, definition_default: List[T] = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize List annotation. """
        return List[TypingProperty._args_from_serialized(serialized)]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write HomogeneousSequence as a Dict. """
        if not title:
            title = 'Items'
        chunk = super().to_dict(title=title, editable=editable, description=description)
        items = [sp.to_dict(title=f"{title}/{i}", editable=editable) for i, sp in enumerate(self.args_schemas)]
        chunk.update({'type': 'array', "items": items[0]})
        return chunk

    def default_value(self):
        """ Default of a sequence. Always return None as default mutable is prohibited. """
        return None

    def check_list(self) -> CheckList:
        """ Check validity of List Type Hint. """
        issues = super().check_list()
        issues += CheckList([self.has_one_arg(), self.has_no_default()])
        return issues

    def has_no_default(self) -> PassedCheck:
        """ Check if List doesn't define a default value that is other than None. """
        if self.definition_default is not None:
            msg = f"{self.check_prefix}Mutable List input defines a default value other than None," \
                  f"which will lead to unexpected behavior and therefore, is not supported."
            return UnsupportedDefault(msg)
        msg = f"{self.check_prefix}Mutable List doesn't define a default value other than None."
        return PassedCheck(msg)


class DynamicDict(TypingProperty):
    """
    Schema class for Dict type hints.

    Datatype that can be seen as a Dict. Have restricted amount of arguments (one for key, one for values),
    but an unlimited length.
    """

    def __init__(self, annotation: Type[Dict[str, Builtin]], attribute: str,
                 definition_default: Dict[str, Builtin] = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Dict annotation. """
        return Dict[TypingProperty._args_from_serialized(serialized)]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write DynamicDict as a Dict. """
        key_type, value_type = self.args
        if key_type != str:
            # !!! Should we support other types ? Numeric ?
            raise NotImplementedError('Non strings keys not supported')
        if value_type not in TYPING_EQUIVALENCES:
            raise ValueError(f'Dicts should have only builtins keys and values, got {value_type}')
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object',
                      'patternProperties': {
                          '.*': {
                            'type': TYPING_EQUIVALENCES[value_type]
                          }
                      }})
        return chunk

    def default_value(self):
        """ Default of a dynamic Dict. Always return None as default mutable is prohibited. """
        return None

    def check_list(self) -> CheckList:
        """ Check validity of DynamicDict Type Hint. """
        issues = super().check_list()
        checks = [self.has_two_args(), self.has_string_keys(), self.has_simple_values(), self.has_no_default()]
        issues += CheckList(checks)
        return issues

    def has_two_args(self) -> PassedCheck:
        """ Annotation should have exactly two arguments, first one for keys, second one for values. """
        if len(self.args) != 2:
            msg = f"{self.check_prefix}is typed as a 'Dict' which requires exactly 2 arguments. " \
                  f"Expected 'Dict[KeyType, ValueType]', got '{self.annotation}'."
            return WrongNumberOfArguments(msg)
        return PassedCheck(f"{self.check_prefix}has two args in its definition : '{self.annotation}'.")

    def has_string_keys(self):
        """ Key Type should be str. """
        key_type, value_type = self.args
        if not issubclass(key_type, str):
            # Should we support other types ? Numeric ?
            msg = f"{self.check_prefix}is typed as a 'Dict[{key_type}, {value_type}]' " \
                  f"which requires str as its key type. Expected 'Dict[str, ValueType]', got '{self.annotation}'."
            return WrongType(msg)
        return PassedCheck(f"{self.check_prefix}has str keys : '{self.annotation}'.")

    def has_simple_values(self):
        """ Value Type should be simple. """
        key_type, value_type = self.args
        if value_type not in TYPING_EQUIVALENCES:
            msg = f"{self.check_prefix}is typed as a 'Dict[{key_type}, {value_type}]' " \
                  f"which requires a builtin type as its value type. " \
                  f"Expected 'int', 'float', 'bool' or 'str', got '{value_type}'."
            return WrongType(msg)
        return PassedCheck(f"{self.check_prefix}has simple values : '{self.annotation}'.")

    def has_no_default(self) -> PassedCheck:
        """ Check if Dict doesn't define a default value that is other than None. """
        if self.definition_default is not None:
            msg = f"{self.check_prefix}Mutable Dict input defines a default value other than None," \
                  f"which will lead to unexpected behavior and therefore, is not supported."
            return UnsupportedDefault(msg)
        msg = f"{self.check_prefix}Mutable Dict doesn't define a default value other than None."
        return PassedCheck(msg)


BaseClass = TypeVar("BaseClass", bound=CoreDessiaObject)


class InstanceOfProperty(TypingProperty):
    """
    Schema class for InstanceOf type hints.

    Datatype that can be seen as a union of classes that inherits from the only arg given.
    Instances of these classes validate against this type.
    """

    def __init__(self, annotation: Type[InstanceOf[BaseClass]], attribute: str,
                 definition_default: BaseClass = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize InstanceOf annotation. """
        return InstanceOf[TypingProperty._args_from_serialized(serialized)]

    @property
    def schema(self):
        """ Get Schema of base class. """
        return ClassSchema(self.args[0])

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write InstanceOf as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        class_ = self.args[0]
        chunk.update({'type': 'object', 'instance_of': self.serialized, 'standalone_in_db': class_._standalone_in_db})
        return chunk

    def default_value(self) -> BaseClass:
        """ Default value of an object. """
        return object_default(definition_default=self.definition_default, class_schema=self.schema)

    def check_list(self) -> CheckList:
        """ Check validity of InstanceOf Type Hint. """
        issues = super().check_list()
        issues += CheckList([self.has_one_arg()])
        return issues


class SubclassProperty(TypingProperty):
    """
    Schema class for Subclass type hints.

    Datatype that can be seen as a union of classes that inherits from the only arg given.
    Classes validate against this type.
    """

    def __init__(self, annotation: Type[Subclass[BaseClass]], attribute: str,
                 definition_default: Type[BaseClass] = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Subclass annotation. """
        return Subclass[TypingProperty._args_from_serialized(serialized)]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Subclass as a Dict. """
        raise NotImplementedError("Subclass is not implemented yet")

    def check_list(self) -> CheckList:
        """
        Check validity of Subclass Type Hint.

        Checks performed :
        - Annotation has exactly one argument, which is the type of the base class.
        """
        issues = super().check_list()
        issues += CheckList([self.has_one_arg()])
        return issues


class MethodTypeProperty(TypingProperty):
    """
    Schema class for MethodType and ClassMethodType type hints.

    A specifically instantiated MethodType validated against this type.
    """

    def __init__(self, annotation: Type[MethodType], attribute: str, definition_default: MethodType = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

        self.class_ = self.args[0]
        self.class_schema = get_schema(annotation=self.class_, attribute=attribute,
                                       definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Methods annotation. Support Class and Instance methods. """
        type_ = TypingProperty.type_from_serialized(serialized)
        if type_ == "MethodType":
            return MethodType[TypingProperty._args_from_serialized(serialized)]
        return ClassMethodType[TypingProperty._args_from_serialized(serialized)]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write MethodType as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        classmethod_ = self.origin is ClassMethodType
        chunk.update({
            'type': 'object', 'is_method': True, 'classmethod_': classmethod_,
            'properties': {
                'class_': self.class_schema.to_dict(title=title, editable=editable, description=description),
                'name': {
                    'type': 'string'
                }
            }
        })
        return chunk

    def check_list(self) -> CheckList:
        """
        Check validity of MethodType Type Hint.

        Checks performed :
        - Class has method TODO
        """
        return CheckList([])


Class = TypeVar("Class", bound=type)


class ClassProperty(TypingProperty):
    """
    Schema class for 'Type' type hints.

    Non DessiaObject sub-classes validated against this type.
    """

    def __init__(self, annotation: Type[Class], attribute: str, definition_default: Class = None):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Type annotation. Support undefined and defined arg. """
        args = TypingProperty._args_from_serialized(serialized)
        if args:
            return Type[args]
        return Type

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Class as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}})
        return chunk

    def check_list(self) -> CheckList:
        """
        Check validity of Class Type Hint.

        Checks performed :
        - Annotation has exactly 1 argument
        """
        issues = super().check_list()
        issues += CheckList([self.has_one_arg()])
        return issues


class GenericTypeProperty(Property):
    """ Meta Property for Types. """

    def __init__(self, annotation: Type[TypeVar], attribute: str, definition_default: TypeVar):
        super().__init__(annotation=annotation, attribute=attribute, definition_default=definition_default)

    @classmethod
    def annotation_from_serialized(cls, serialized: str):
        """ Deserialize Generic Type annotation. """
        raise NotImplementedError("Annotation deserialization not implemented for Generic Types")

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Type as a Dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}})
        return chunk


class AnyProperty(Property):
    """ Handle Any typed (cannot be form inputs). """

    def __init__(self, attribute: str):
        super().__init__(annotation=Any, attribute=attribute, definition_default=None)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({"properties": {".*": ".*"}, "type": "object"})
        return chunk


def inspect_arguments(method: Callable, merge: bool = False):
    """ Wrapper around 'split_default_argument' method in order to call it from a method object. """
    method_full_name = f'{method.__module__}.{method.__qualname__}'
    if method_full_name in _fullargsspec_cache:
        argspecs = _fullargsspec_cache[method_full_name]
    else:
        argspecs = inspect.getfullargspec(method)
        _fullargsspec_cache[method_full_name] = argspecs
    return split_default_args(argspecs=argspecs, merge=merge)


def split_default_args(argspecs: inspect.FullArgSpec, merge: bool = False):
    """
    Find default value and required arguments of class construction.

    Get method arguments and default arguments as sequences while removing forbidden ones (self, cls...).
    """
    nargs, ndefault_args = split_argspecs(argspecs)

    default_arguments = {}
    arguments = []
    for iargument, argument in enumerate(argspecs.args[1:]):
        if argument not in RESERVED_ARGNAMES:
            if iargument >= nargs - ndefault_args:
                default_value = argspecs.defaults[ndefault_args - nargs + iargument]
                if merge:
                    arguments.append((argument, default_value))
                else:
                    default_arguments[argument] = default_value
            else:
                arguments.append(argument)
    return arguments, default_arguments


def split_argspecs(argspecs: inspect.FullArgSpec) -> Tuple[int, int]:
    """ Get number of regular arguments as well as arguments with default values. """
    nargs = len(argspecs.args) - 1
    if argspecs.defaults is not None:
        ndefault_args = len(argspecs.defaults)
    else:
        ndefault_args = 0
    return nargs, ndefault_args


def get_schema(annotation: Type[T], attribute: str = "", definition_default: Optional[T] = None) -> Property:
    """ Get schema Property object from given annotation. """
    if annotation is None or inspect.isclass(annotation) and issubclass(annotation, type(None)):
        return GenericTypeProperty(annotation=annotation, attribute=attribute, definition_default=definition_default)
    if annotation in TYPING_EQUIVALENCES:
        return BuiltinProperty(annotation=annotation, attribute=attribute, definition_default=definition_default)
    if is_typing(annotation):
        return typing_schema(typing_=annotation, attribute=attribute, definition_default=definition_default)
    if hasattr(annotation, '__origin__') and annotation.__origin__ is type:
        # Type is not considered a Typing as it has no arguments
        return ClassProperty(annotation=annotation, attribute=attribute, definition_default=definition_default)
    if annotation is Any:
        return AnyProperty(attribute)
    if inspect.isclass(annotation):
        return custom_class_schema(annotation=annotation, attribute=attribute, definition_default=definition_default)
    if isinstance(annotation, TypeVar):
        return GenericTypeProperty(annotation=annotation, attribute=attribute, definition_default=definition_default)
    raise NotImplementedError(f"No schema defined for attribute '{attribute}' annotated '{annotation}'.")


ORIGIN_TO_SCHEMA_CLASS = {
    tuple: HeterogeneousSequence, list: HomogeneousSequence, collections.abc.Iterator: HomogeneousSequence,
    Union: UnionProperty, dict: DynamicDict, InstanceOf: InstanceOfProperty,
    MethodType: MethodTypeProperty, ClassMethodType: MethodTypeProperty, type: ClassProperty
}

SERIALIZED_TO_SCHEMA_CLASS = {
    "int": BuiltinProperty, "float": BuiltinProperty, "bool": BuiltinProperty, "str": BuiltinProperty,
    "Tuple": HeterogeneousSequence, "List": HomogeneousSequence, "Iterator": HomogeneousSequence,
    "Union": UnionProperty, "Dict": DynamicDict, "InstanceOf": InstanceOfProperty, "Subclass": SubclassProperty,
    "MethodType": MethodTypeProperty, "ClassMethodType": MethodTypeProperty, "Type": ClassProperty
}


def serialize_annotation(annotation: Type[T], attribute: str = "") -> str:
    """ Make use of schema to serialized annotations. """
    schema = get_schema(annotation=annotation, attribute=attribute)
    return schema.serialized


def deserialize_annotation(serialized: str) -> Type[T]:
    """ From a string denoting an annotation, get deserialize value. """
    if "[" in serialized:
        return TypingProperty.annotation_from_serialized(serialized)
    if serialized in SERIALIZED_TO_SCHEMA_CLASS:
        return SERIALIZED_TO_SCHEMA_CLASS[serialized].annotation_from_serialized(serialized)
    return get_python_class_from_class_name(serialized)


def is_typing(object_) -> bool:
    """ Return True if given object can be seen as a typing (has a module, an origin and arguments). """
    has_module = hasattr(object_, '__module__')
    has_origin = hasattr(object_, '__origin__')
    has_args = hasattr(object_, '__args__')
    return has_module and has_origin and has_args


def typing_schema(typing_: Type[T], attribute: str, definition_default: T = None) -> Property:
    """ Get schema Property for typing annotations. """
    origin = get_origin(typing_)
    if origin is Union and union_is_default_value(typing_):
        # This is a false UnionProperty => Is a default value set to None
        return OptionalProperty(annotation=typing_, attribute=attribute, definition_default=definition_default)
    try:
        return ORIGIN_TO_SCHEMA_CLASS[origin](typing_, attribute, definition_default)
    except KeyError as exc:
        raise NotImplementedError(f"No Schema defined for typing '{typing_}'.") from exc


def custom_class_schema(annotation: Type[T], attribute: str, definition_default: T = None) -> Property:
    """ Get schema Property object for non typing annotations. """
    if issubclass(annotation, Measure):
        return MeasureProperty(annotation=annotation, attribute=attribute, definition_default=definition_default)
    if issubclass(annotation, (BinaryFile, StringFile)):
        return FileProperty(annotation=annotation, attribute=attribute, definition_default=definition_default)
    if issubclass(annotation, CoreDessiaObject):
        # Dessia custom classes
        return CustomClass(annotation=annotation, attribute=attribute, definition_default=definition_default)
    raise NotImplementedError(f"No Schema defined for type '{annotation}'.")


def object_default(definition_default: CoreDessiaObject = None, class_schema: ClassSchema = None):
    """
    Default value of an object.

    Return serialized user default if definition, else None.
    """
    if definition_default is not None:
        return definition_default.to_dict(use_pointers=False)
    if class_schema is not None:
        # TODO Should we implement this ? Right now, tests state that the result is None
        # return class_schema.default_dict()
        pass
    return None


def compute_typing_schema_serialization(serialized_typing: str, args_schemas: List[Property]) -> str:
    """ Build final typing serialized string. """
    return f"{serialized_typing}[{', '.join([s.serialized for s in args_schemas])}]"


def serialize_typing(typing_, attribute: str = "") -> str:
    """ Make use of schema to serialized annotations. """
    warnings.warn("Function serialize_typing have been renamed serialize_annotation. Please use it instead.",
                  DeprecationWarning)
    return serialize_annotation(annotation=typing_, attribute=attribute)


def union_is_default_value(typing_: Type) -> bool:
    """
    Union typings can be False positives.

    An argument of a function that has a default_value set to None is Optional[T], which is an alias for
    Union[T, NoneType]. This function checks if this is the case.
    """
    args = get_args(typing_)
    return len(args) == 2 and type(None) in args


def extract_args(string: str) -> List[str]:
    """
    Extract first level arguments from serialized typing.

    This function does not split by commas when we are down to second level annotations
    and has been preferred to regexp for this reason.
    """
    opened_brackets = 0
    closed_brackets = 0
    current_arg = ""
    arguments = []
    for character in string.replace(" ", ""):
        if character == "[":
            opened_brackets += 1
        if character == "]":
            closed_brackets += 1
        split_by_comma = closed_brackets == opened_brackets
        if split_by_comma and character == ",":
            # We are at first level, because we closed as much brackets as we have opened
            # Current argument is complete, we append it to args sequence and reset current_args
            arguments.append(current_arg)
            current_arg = ""
        else:
            # We are at a deeper level, because all opened brackets haven't been closed.
            # We build current argument
            current_arg += character
    if current_arg:
        # Append last argument to arguments sequence
        arguments.append(current_arg)
    return arguments


class ParsedAttribute(TypedDict):
    """ Parsed description of a docstring attribute. """

    desc: str
    type_: str
    annotation: str


class ParsedDocstring(TypedDict):
    """ Parsed description of a docstring. """

    description: str
    attributes: Dict[str, ParsedAttribute]


def parse_class_docstring(class_) -> ParsedDocstring:
    """ Helper to get parse docstring from a class. """
    docstring = class_.__doc__
    annotations = get_type_hints(class_.__init__)
    return parse_docstring(docstring=docstring, annotations=annotations)


def parse_docstring(docstring: str, annotations: Dict[str, Any]) -> ParsedDocstring:
    """ Parse user-defined docstring of given class. Refer to docs to see how docstrings should be built. """
    if docstring:
        no_return_docstring = docstring.split(':return:')[0]
        splitted_docstring = no_return_docstring.split(':param ')
        parsed_docstring = {"description": splitted_docstring[0].strip()}
        params = splitted_docstring[1:]
        args = {}
        for param in params:
            argname, parsed_attribute = parse_attribute(param, annotations)
            args[argname] = parsed_attribute
        parsed_docstring.update({'attributes': args})
        return parsed_docstring
    return {'description': "", 'attributes': {}}


def parse_attribute(param, annotations) -> Tuple[str, ParsedAttribute]:
    """ Extract attribute from user-defined docstring. """
    if ":type" in param:
        param = param.split(":type ")[0]
    argname, argdesc = param.split(":", maxsplit=1)
    annotation = annotations[argname]
    parsed_attribute = {'desc': argdesc.strip(), 'type_': serialize_annotation(annotation),
                        'annotation': str(annotation)}
    return argname, parsed_attribute


EMPTY_PARSED_ATTRIBUTE = {"desc": "", "type": "", "annotation": ""}
FAILED_DOCSTRING_PARSING = {'description': 'Docstring parsing failed', 'attributes': {}}
FAILED_ATTRIBUTE_PARSING = {"desc": 'Attribute documentation parsing failed',
                            "type": "", "annotation": ""}


def _check_docstring(element):
    """ Return True if an object, a class or a method have a proper docstring. Otherwise, return False. """
    docstring = element.__doc__
    if docstring is None:
        print(f'Docstring not found for {element}')
        return False
    if inspect.isclass(element):
        # element is an object or a class
        annotations = get_type_hints(element.__init__)
    elif inspect.ismethod(element) or inspect.isfunction(element):
        # element is a method
        annotations = get_type_hints(element)
    else:
        raise NotImplementedError
    try:
        parse_docstring(docstring=docstring, annotations=annotations)
        return True
    except TypeError:
        return False