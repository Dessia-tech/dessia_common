from typing import TypeVar, Type, Generic

T = TypeVar('T')

# Subclass = Type[T]


class Subclass(Generic[T]):
    Generic.__init__(_name='Subclass')
    pass
