from typing import TypeVar, Generic, Dict, Any, Tuple


T = TypeVar('T')


class Subclass(Generic[T]):
    pass


class InstanceOf(Generic[T]):
    pass


# Types Aliases
JsonSerializable = Dict[str, Any]
RGBColor = Tuple[float, float, float]


# Measures
class Measure(float):
    units = ''

    
class Distance(Measure):
    units = 'm'


class Mass(Measure):
    units = 'kg'


class Force(Measure):
    units = 'N'


class Torque(Measure):
    units = 'Nm'


class Stress(Measure):
    units = 'Pa'
