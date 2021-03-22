from typing import TypeVar, Generic, Dict, Any, Tuple
import warnings


T = TypeVar('T')


class Subclass(Generic[T]):
    warnings.warn("Subclass is deprecated, use InstanceOf instead",
                  DeprecationWarning)
    pass


# Types Aliases
JsonSerializable = Dict[str, Any]
RGBColor = Tuple[float, float, float]


# Measures
class Measure(float):
    units = ''


class Distance(Measure):
    units = 'm'
