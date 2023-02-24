""" Typing for dessia_common. """
from typing import TypeVar, Generic, Dict, Any, Tuple

import dessia_common.typings
from dessia_common.utils.helpers import full_classname, get_python_class_from_class_name


T = TypeVar('T')


class Subclass(Generic[T]):
    """ Typing that denotes a Subclass of T. """


class InstanceOf(Generic[T]):
    """ Typing that denotes a Instance of T. """


class MethodType(Generic[T]):
    """ Typing that denotes a method of class T. """

    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name

    def __deepcopy__(self, memo=None):
        return MethodType(self.class_, self.name)

    def get_method(self):
        """ Helper to get real method from class_ and method name. """
        return getattr(self.class_, self.name)

    def to_dict(self):
        classname = full_classname(object_=self.class_, compute_for='class')
        method_type_classname = full_classname(object_=self.__class__, compute_for="class")
        return {"class_": classname, "name": self.name, "object_class": method_type_classname}

    @classmethod
    def dict_to_object(cls, dict_) -> 'MethodType':
        class_ = get_python_class_from_class_name(dict_["class_"])
        return cls(class_=class_, name=dict_["name"])


class ClassMethodType(MethodType[T]):
    """ Typing that denotes a classmethod of class T. """

    def __init__(self, class_: T, name: str):
        MethodType.__init__(self, class_=class_, name=name)


class AttributeType(Generic[T]):
    """ Typing that denotes an attribute of class T. """

    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name


class ClassAttributeType(AttributeType[T]):
    """ Typing that denotes a class attribute of class T. """

    def __init__(self, class_: T, name: str):
        AttributeType.__init__(self, class_=class_, name=name)


# Types Aliases
JsonSerializable = Dict[str, Any]
RGBColor = Tuple[float, float, float]
