from typing import TypeVar, Generic, Dict, Any

T = TypeVar('T')


class Subclass(Generic[T]):
    pass

# Types Aliases
JsonSerializable = Dict[str, Any]
