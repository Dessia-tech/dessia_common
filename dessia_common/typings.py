"""
typings for dessia_common

"""
from typing import TypeVar, Generic, Dict, Any, Tuple


T = TypeVar('T')


class Subclass(Generic[T]):
    pass
#     def __new__(cls, *args, **kwargs) -> T:
#         raise NotImplementedError("Should not try to instantiate Subclass")


class InstanceOf(Generic[T]):
    pass
#     def __new__(cls, *args, **kwargs) -> T:
#         raise NotImplementedError("Should not try to instantiate InstanceOf")


class MethodType(Generic[T]):
    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name

    def __deepcopy__(self, memo=None):
        return MethodType(self.class_, self.name)

    def get_method(self):
        return getattr(self.class_, self.name)


class ClassMethodType(MethodType[T]):
    def __init__(self, class_: T, name: str):
        MethodType.__init__(self, class_=class_, name=name)


class AttributeType(Generic[T]):
    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name


class ClassAttributeType(AttributeType[T]):
    def __init__(self, class_: T, name: str):
        AttributeType.__init__(self, class_=class_, name=name)


# Types Aliases
JsonSerializable = Dict[str, Any]
RGBColor = Tuple[float, float, float]
