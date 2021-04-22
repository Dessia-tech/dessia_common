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
    
class Time(Measure):
    units = 's'

class Speed(Measure):
    units = 'm/s'

class Acceleration(Measure):
    units = 'm/s²'

class Mass(Measure):
    units ='Kg'

class Force(Measure):
    units = 'N'

class Work(Measure):
    units = 'N*m'
    
class Ampere(Measure):
    units = 'A'
    

