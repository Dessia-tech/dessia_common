""" Typings for dessia_common. """
from typing import TypeVar, Generic, Dict, Any, Tuple


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
