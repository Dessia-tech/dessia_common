"""
typings for dessia_common

"""
from typing import TypeVar, Generic, Dict, Any, Tuple


T = TypeVar('T')


class Subclass(Generic[T]):
    pass


class InstanceOf(Generic[T]):
    pass


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


# Measures
class Measure(float):
    units = ''


class Distance(Measure):
    units = 'm'


class Angle(Measure):
    units = 'radians'


class Torque(Measure):
    units = 'Nm'


class Stress(Measure):
    units = 'Pa'


class Time(Measure):
    units = 's'


class Speed(Measure):
    units = 'm/s'


class Acceleration(Measure):
    units = 'm/s²'


class Mass(Measure):
    units = 'Kg'


class Force(Measure):
    units = 'N'


class Work(Measure):
    units = 'N*m'


class Power(Measure):
    units = 'N*m/s'
