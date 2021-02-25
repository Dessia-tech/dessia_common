from typing import TypeVar, Generic, Dict, Any, List, Tuple

T = TypeVar('T')


class Subclass(Generic[T]):
    pass

# Types Aliases
JsonSerializable = Dict[str, Any]

RGBColor = Tuple[float, float, float]