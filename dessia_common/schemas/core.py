#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema generation functions
"""
from copy import deepcopy
import inspect
import collections.abc
import typing as tp
import dessia_common.utils.types as dc_types
from dessia_common.abstract import CoreDessiaObject
from dessia_common.files import BinaryFile, StringFile
from dessia_common.typings import Subclass, MethodType, ClassMethodType, Any, InstanceOf
from dessia_common.measures import Measure
from dessia_common.utils.docstrings import parse_docstring, FAILED_DOCSTRING_PARSING, FAILED_ATTRIBUTE_PARSING
from dessia_common.utils.helpers import prettyname
from dessia_common.schemas.interfaces import Annotations
from dessia_common.checks import CheckList, FailedCheck, PassedCheck

SCHEMA_HEADER = {"definitions": {}, "$schema": "http://json-schema.org/d_raft-07/schema#",
                 "type": "object", "required": [], "properties": {}}
RESERVED_ARGNAMES = ['self', 'cls', 'progress_callback', 'return']

_fullargsspec_cache = {}


class Schema:
    """
    Abstraction of a Schema.

    It reads the user-defined type hints and then writes into a dict the recursive structure of an object
    that can be handled by dessia_common.
    This dictionnary can then be translated as a json to be read by the frontend in order to compute edit forms,
    for example.

    Right now Schema doesn't inherit from any DessiaObject class (SerializableObject ?), but could, in the future.
    That is why it implements methods with the same name.
    """

    def __init__(self, annotations: Annotations, argspec: inspect.FullArgSpec, docstring: str):
        self.annotations = annotations
        self.attributes = [a for a in argspec.args if a not in RESERVED_ARGNAMES]

        self.property_schemas = {a: get_schema(annotations[a]) for a in self.attributes}

        try:  # Parse docstring
            self.parsed_docstring = parse_docstring(docstring=docstring, annotations=annotations)
        except Exception:  # Catching broad exception because we don't want it to break platform if a failure occurs
            self.parsed_docstring = FAILED_DOCSTRING_PARSING

        self.parsed_attributes = self.parsed_docstring['attributes']

        self.required_arguments, self.default_arguments = split_default_args(argspecs=argspec, merge=False)

        self.check_list().raise_if_above_level("error")

    @property
    def editable_attributes(self):
        """ Attributes that are not in RESERVED_ARGNAMES. """
        return [a for a in self.attributes if a not in RESERVED_ARGNAMES]

    def chunk(self, attribute: str):
        """ Extract and compute a schema from one of the attributes. """
        schema = self.property_schemas[attribute]

        if self.parsed_attributes is not None and attribute in self.parsed_attributes:
            try:
                description = self.parsed_attributes[attribute]['desc']
            except Exception:  # Catching broad exception because we don't want it to break platform if a failure occurs
                description = FAILED_ATTRIBUTE_PARSING["desc"]
        else:
            description = ""

        editable = attribute in self.editable_attributes
        chunk = schema.to_dict(title=prettyname(attribute), editable=editable, description=description)

        if attribute in self.default_arguments:
            # TODO Could use this and Optional proxy in order to inject real default values for mutables
            chunk = set_default_value(schema_element=chunk, default_value=self.default_arguments[attribute])
        return chunk

    @property
    def chunks(self):
        """ Concatenate schema chunks into a list. """
        return [self.chunk(a) for a in self.attributes]

    def to_dict(self):
        """ Write the whole schema. """
        schema = deepcopy(SCHEMA_HEADER)
        properties = {a: self.chunk(a) for a in self.attributes}
        schema.update({"required": self.required_arguments, "properties": properties,
                       "description": self.parsed_docstring["description"]})
        return schema

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self) -> CheckList:
        """
        Browse all properties and list potential issues.

        Checks performed for each argument :
        - Is typed in method definition
        - Schema specific check
        """
        issues = CheckList([])

        for attribute in self.attributes:
            # Is typed
            issues += self.attribute_is_annotated(attribute)

            # Specific check
            schema = self.property_schemas[attribute]
            issues += schema.check_list(attribute)
        return issues

    def is_valid(self) -> bool:
        """ Return wether the class definition is valid or not. """
        return self.check_list().checks_above_level("error")

    def attribute_is_annotated(self, attribute: str) -> PassedCheck:
        """ Check whether given attribute is annotated in function definition or not. """
        if attribute not in self.annotations:
            return FailedCheck(f"Attribute {attribute} has no typing")
        return PassedCheck(f"Attribute '{attribute}' is annotated")


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
        """ Attributes that are not in RESERVED_ARGNAMES nor defined as non editable by user. """
        attributes = super().editable_attributes
        return [a for a in attributes if a not in self.class_._non_editable_attributes]

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")


class MethodSchema(Schema):
    """
    Schema of a method.

    Given method should be one of a DessiaObject. It reads its annotations.
    """
    def __init__(self, method: tp.Callable):
        self.method = method

        annotations = tp.get_type_hints(method)
        members = inspect.getfullargspec(method)
        docstring = method.__doc__
        Schema.__init__(self, annotations=annotations, argspec=members, docstring=docstring)

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")


class Property:
    """ Base class for a schema property. """
    def __init__(self, annotation: tp.Type):
        self.annotation = annotation

    @property
    def schema(self):
        """ Return a reference to itself. Might be overwritten for proxy such as Optional or Annotated. """
        return self

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write base schema as a dict. """
        return {'title': title, 'editable': editable, 'description': description,
                'python_typing': dc_types.serialize_typing(self.annotation), "type": None}

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, _: str) -> CheckList:
        """
        Check validity of Property Type Hint.

        Checks performed : None. TODO ?
        """
        return CheckList([])


class TypingProperty(Property):
    """ Schema class for typing based annotations. """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    @property
    def args(self):
        """ Return Typing arguments. """
        return tp.get_args(self.annotation)

    @property
    def origin(self):
        """ Return Typing origin. """
        return tp.get_origin(self.annotation)

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def has_one_arg(self, attribute: str) -> PassedCheck:
        """ Annotation should have exactly one argument; """
        if len(self.args) != 1:
            pretty_origin = prettyname(self.origin.__name__)
            msg = f"Argument '{attribute}' is typed as a '{pretty_origin}' which requires exactly 1 argument. " \
                  f"Expected '{pretty_origin}[T]', got '{self.annotation}'."
            return FailedCheck(msg)
        return PassedCheck(f"Argument '{attribute}' has exactly one arg in its definition.")


class OptionalProperty(TypingProperty):
    """
    Proxy Schema class for OptionalProperty properties.

    OptionalProperty is only a catch for arguments that default to None.
    Arguments with default values other than None are not considered Optionals
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    @property
    def schema(self):
        """ Return a reference to its only arg. """
        return get_schema(self.args[0])

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Optional as a dict. """
        default_value = None
        chunk = self.schema.to_dict(title=title, editable=editable, description=description)
        chunk["default_value"] = default_value
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")


class AnnotatedProperty(TypingProperty):
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
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)
        raise NotImplementedError(self._not_implemented_msg)

    @property
    def schema(self):
        """ Return a reference to its only arg. """
        return get_schema(self.args[0])

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Annotated as a dict. """
        raise NotImplementedError(self._not_implemented_msg)

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """
        Check validity of DynamicDict Type Hint.

        Checks performed : None. TODO : Arg validity
        """
        raise NotImplementedError(self._not_implemented_msg)


class BuiltinProperty(Property):
    """ Schema class for Builtin type hints. """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Builtin as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk["type"] = dc_types.TYPING_EQUIVALENCES[self.annotation]
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")


class MeasureProperty(BuiltinProperty):
    """ Schema class for Measure type hints. """
    def __init__(self, annotation: tp.Type[Measure]):
        super().__init__(annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Measure as a dict. """
        chunk = Property.to_dict(self, title=title, editable=editable, description=description)
        chunk.update({"si_unit": self.annotation.si_unit, "type": "number"})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")


class FileProperty(Property):
    """ Schema class for File type hints. """
    def __init__(self, annotation: tp.Type):
        Property.__init__(self, annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write File as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'text', 'is_file': True})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")


class CustomClass(Property):
    """ Schema class for CustomClass type hints. """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)
        self.classname = dc_types.full_classname(object_=self.annotation, compute_for='class')

    @property
    def schema(self):
        """ Return a reference to the schema of the annotation. """
        return ClassSchema(self.annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write CustomClass as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'standalone_in_db': self.annotation._standalone_in_db,
                      "classes": [self.classname]})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """
        Check validity of user custom class Type Hint.

        Checks performed :
        - Is subclass of DessiaObject
        """
        issues = super().check_list(attribute)
        issues += CheckList([self.is_dessia_object_typed(attribute)])
        return issues

    def is_dessia_object_typed(self, attribute: str) -> PassedCheck:
        """ Check whether if typing for given attribute annotates a subclass of DessiaObject or not . """
        if not issubclass(self.annotation, CoreDessiaObject):
            return FailedCheck(f"Attribute '{attribute}' : Class '{self.classname}' is not a subclass of DessiaObject")
        msg = f"Attribute '{attribute}' : Class '{self.classname}' is properly typed as a subclass of DessiaObject"
        return PassedCheck(msg)


class UnionProperty(TypingProperty):
    """ Schema class for Union type hints. """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        standalone_args = [a._standalone_in_db for a in self.args]
        if all(standalone_args):
            self.standalone = True
        elif not any(standalone_args):
            self.standalone = False
        else:
            self.standalone = None

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Union as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        classnames = [dc_types.full_classname(object_=a, compute_for='class') for a in self.args]
        chunk.update({'type': 'object', 'classes': classnames, 'standalone_in_db': self.standalone})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """
        Check validity of UnionProperty Type Hint.

        Checks performed :
        - Subobject are all standalone or none of them are. TODO : What happen if args are not DessiaObjects ?
        """
        issues = super().check_list(attribute)
        issues += CheckList([self.classes_are_standalone_consistent(attribute)])
        return issues

    def classes_are_standalone_consistent(self, attribute: str) -> PassedCheck:
        """ Check whether all class in Union are standalone or none of them are. """
        standalone_args = [a._standalone_in_db for a in self.args]
        if all(standalone_args):
            msg = f"Attribute '{attribute}' : All arguments of Union type '{self.annotation}' are standalone in db."
            return PassedCheck(msg)
        if not any(standalone_args):
            msg = f"Attribute '{attribute}' : No arguments of Union type '{self.annotation}' are standalone in db."
            return PassedCheck(msg)
        msg = f"Attribute '{attribute}' : 'standalone_in_db' values for arguments of Union type '{self.annotation}'" \
              f"are not consistent. They should be all standalone in db or none of them should."
        return FailedCheck(msg)


class HeterogeneousSequence(TypingProperty):
    """
    Schema class for Tuple type hints.

    Datatype that can be seen as a tuple. Have any amount of arguments but a limited length.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        self.item_schemas = [get_schema(a) for a in self.args]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write HeterogeneousSequence as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        items = [sp.to_dict() for sp in self.item_schemas]
        chunk.update({'type': 'array', 'additionalItems': False, 'items': items})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """ Check validity of Tuple Type Hint. """
        issues = super().check_list(attribute)
        issues += CheckList([self.has_enough_args(attribute)])
        return issues

    def has_enough_args(self, attribute) -> PassedCheck:
        """ Annotation should have at least one argument, one for each element of the tuple. """
        if len(self.args) == 0:
            msg = f"Attribute '{attribute}' is typed as a 'Tuple' which requires at least 1 argument. " \
                  f"Expected 'Tuple[T0, T1, ..., Tn]', got '{self.annotation}'."
            return FailedCheck(msg)
        return PassedCheck(f"Attribute '{attribute}' has several arguments : '{self.annotation}'")


class HomogeneousSequence(TypingProperty):
    """
    Schema class for List type hints.

    Datatype that can be seen as a list. Have only one arguments but an unlimited length.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        self.item_schemas = [get_schema(a) for a in self.args]

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write HomogeneousSequence as a dict. """
        if not title:
            title = 'Items'
        chunk = super().to_dict(title=title, editable=editable, description=description)
        items = [sp.to_dict(title=title, editable=editable, description=description) for sp in self.item_schemas]
        chunk.update({'type': 'array', 'python_typing': dc_types.serialize_typing(self.annotation),
                      "items": items[0]})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """ Check validity of List Type Hint. """
        issues = super().check_list(attribute)
        issues += CheckList([self.has_one_arg(attribute)])
        return issues


class DynamicDict(TypingProperty):
    """
    Schema class for Dict type hints.

    Datatype that can be seen as a dict. Have restricted amount of arguments (one for key, one for values),
    but an unlimited length.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write DynamicDict as a dict. """
        key_type, value_type = self.args
        if key_type != str:
            # !!! Should we support other types ? Numeric ?
            raise NotImplementedError('Non strings keys not supported')
        if value_type not in dc_types.TYPING_EQUIVALENCES:
            raise ValueError(f'Dicts should have only builtins keys and values, got {value_type}')
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object',
                      'patternProperties': {
                          '.*': {
                            'type': dc_types.TYPING_EQUIVALENCES[value_type]
                          }
                      }})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """ Check validity of DynamicDict Type Hint. """
        issues = super().check_list(attribute)
        checks = [self.has_two_args(attribute), self.has_string_keys(attribute), self.has_string_keys(attribute)]
        issues += CheckList(checks)
        return issues

    def has_two_args(self, attribute: str) -> PassedCheck:
        """ Annotation should have exactly two arguments, first one for keys, second one for values"""
        if len(self.args) != 2:
            msg = f"Argument '{attribute}' is typed as a 'Dict' which requires exactly 2 arguments. " \
                  f"Expected 'Dict[KeyType, ValueType]', got '{self.annotation}'."
            return FailedCheck(msg)
        return PassedCheck(f"Attribute '{attribute}' has two args in its defintion : '{self.annotation}'.")

    def has_string_keys(self, attribute):
        """ Key Type should be str"""
        key_type, value_type = self.args
        if not isinstance(key_type, str):
            # Should we support other types ? Numeric ?
            msg = f"Argument '{attribute}' is typed as a 'Dict[{key_type}, {value_type}]' " \
                  f"which requires str as its key type. Expected 'Dict[str, ValueType]', got '{self.annotation}'."
            return FailedCheck(msg)
        return PassedCheck(f"Attribute '{attribute}' has str keys : '{self.annotation}'.")

    def has_simple_values(self, attribute):
        """ Value Type should be simple. """
        key_type, value_type = self.args
        if value_type not in dc_types.TYPING_EQUIVALENCES:
            msg = f"Argument '{attribute}' is typed as a 'Dict[{key_type}, {value_type}]' " \
                  f"which requires a builtin type as its value type. " \
                  f"Expected 'int', 'float', 'bool' or 'str', got '{value_type}'"
            return FailedCheck(msg)
        return PassedCheck(f"Attribute '{attribute}' has simple values : '{self.annotation}'")


class InstanceOfProperty(TypingProperty):
    """
    Schema class for InstanceOf type hints.

    Datatype that can be seen as an union of classes that inherits from the only arg given.
    Instances of these classes validate against this type.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write InstanceOf as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        class_ = self.args[0]
        classname = dc_types.full_classname(object_=class_, compute_for='class')
        chunk.update({'type': 'object', 'instance_of': classname, 'standalone_in_db': class_._standalone_in_db})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """ Check validity of InstanceOf Type Hint. """
        issues = super().check_list(attribute)
        issues += CheckList([self.has_one_arg(attribute)])
        return issues


class SubclassProperty(TypingProperty):
    """
    Schema class for Subclass type hints.

    Datatype that can be seen as an union of classes that inherits from the only arg given.
    Classes validate against this type.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Subclass as a dict. """
        raise NotImplementedError("Subclass is not implemented yet")

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """
        Check validity of Subclass Type Hint.

        Checks performed :
        - Annotation has exactly one argument, which is the type of the base class.
        """
        issues = super().check_list(attribute)
        issues += CheckList([self.has_one_arg(attribute)])
        return issues


class MethodTypeProperty(TypingProperty):
    """
    Schema class for MethodType and ClassMethodType type hints.

    A specifically instantiated MethodType validated against this type.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

        self.class_ = self.args[0]
        self.class_schema = get_schema(self.class_)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write MethodType as a dict. """
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

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """
        Check validity of MethodType Type Hint.

        Checks performed :
        - Class has method TODO
        """
        return CheckList([])


class ClassProperty(TypingProperty):
    """
    Schema class for Type type hints.

    Non DessiaObject subclasses validated against this type.
    """
    def __init__(self, annotation: tp.Type):
        super().__init__(annotation=annotation)

    def to_dict(self, title: str = "", editable: bool = False, description: str = ""):
        """ Write Class as a dict. """
        chunk = super().to_dict(title=title, editable=editable, description=description)
        chunk.update({'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}})
        return chunk

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Build a schema back from its dict representation.

        TODO  Useful for low_code features ?
        """
        raise NotImplementedError("Schema reconstruction is not implemented yet")

    def check_list(self, attribute: str) -> CheckList:
        """
        Check validity of Class Type Hint.

        Checks performed :
        - Annotation has exactly 1 argument
        """
        issues = super().check_list(attribute)
        issues += CheckList([self.has_one_arg(attribute)])
        return issues


def inspect_arguments(method: tp.Callable, merge: bool = False):
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


def split_argspecs(argspecs: inspect.FullArgSpec) -> tp.Tuple[int, int]:
    """ Get number of regular arguments as well as arguments with default values. """
    nargs = len(argspecs.args) - 1

    if argspecs.defaults is not None:
        ndefault_args = len(argspecs.defaults)
    else:
        ndefault_args = 0
    return nargs, ndefault_args


def get_schema(annotation: tp.Type) -> Property:
    """ Get schema Property object from given annotation. """
    if annotation in dc_types.TYPING_EQUIVALENCES:
        return BuiltinProperty(annotation)
    if dc_types.is_typing(annotation):
        return typing_schema(annotation)
    if hasattr(annotation, '__origin__') and annotation.__origin__ is type:
        # TODO Is this deprecated ? Should be used in 3.8 and not 3.9 ?
        # return {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}}}
        pass
    if annotation is Any:
        # TODO Do we still want to support Any ?
        # chunk = {'type': 'object', 'properties': {'.*': '.*'}}
        pass
    if inspect.isclass(annotation):
        return custom_class_schema(annotation)
    raise NotImplementedError(f"No schema defined for annotation '{annotation}'.")


ORIGIN_TO_SCHEMA_CLASS = {
    tuple: HeterogeneousSequence, list: HomogeneousSequence, collections.abc.Iterator: HomogeneousSequence,
    tp.Union: UnionProperty, dict: DynamicDict, InstanceOf: InstanceOfProperty,
    MethodType: MethodTypeProperty, ClassMethodType: MethodTypeProperty, type: ClassProperty
}


def typing_schema(typing_) -> Property:
    """ Get schema Property for typing annotations. """
    origin = tp.get_origin(typing_)
    if origin is tp.Union and dc_types.union_is_default_value(typing_):
        # This is a false UnionProperty => Is a default value set to None
        return OptionalProperty(typing_)
    try:
        return ORIGIN_TO_SCHEMA_CLASS[origin](typing_)
    except KeyError:
        raise NotImplementedError(f"No Schema defined for typing '{typing_}'.")

    # if origin is tp.Union:
    #     if dc_types.union_is_default_value(typing_):
    #         # This is a false UnionProperty => Is a default value set to None
    #         return OptionalProperty(typing_)
    #     # Types union
    #     return UnionProperty(typing_)
    # if origin is tuple:
    #     return HeterogeneousSequence(typing_)
    # if origin in [list, collections.abc.Iterator]:
    #     return HomogeneousSequence(typing_)
    # if origin is dict:
    #     return DynamicDict(typing_)
    # if origin is Subclass:
    #     pass
    # if origin is InstanceOf:
    #     return InstanceOfProperty(typing_)
    # if origin in [MethodType, ClassMethodType]:
    #     return MethodTypeProperty(typing_)
    # if origin is type:
    #     return ClassProperty(typing_)
    # raise NotImplementedError(f"No Schema defined for typing '{typing_}'.")


def custom_class_schema(annotation: tp.Type) -> Property:
    """ Get schema Property object for non typing annotations. """
    if issubclass(annotation, Measure):
        return MeasureProperty(annotation)
    if issubclass(annotation, (BinaryFile, StringFile)):
        return FileProperty(annotation)
    if issubclass(annotation, CoreDessiaObject):
        # Dessia custom classes
        return CustomClass(annotation)
    raise NotImplementedError(f"No Schema defined for type '{annotation}'.")


def default_sequence(array_schema):
    """ Get default value for array schema. """
    if dc_types.is_sequence(array_schema['items']):
        # Tuple schema
        if 'default_value' in array_schema:
            return array_schema['default_value']
        return [default_dict(v) for v in array_schema['items']]
    return None


def datatype_from_schema(schema):
    """ Get datatype from schema. """
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
    """ Get default value from schema. """
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
    """ Get default value for dict. """
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


def set_default_value(schema_element, default_value):
    """ Write default value in jsonschema. """
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
