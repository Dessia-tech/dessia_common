from typing import TypeVar, Generic, Dict, Any, SupportsFloat, Protocol

T = TypeVar('T')

# A = TypeVar('A', int, float, covariant=True)
# D = TypeVar('D', int, float, contravariant=True)


class Subclass(Generic[T]):
    pass


# Types Aliases
JsonSerializable = Dict[str, Any]


# Measures
class Measure(float):
    units = ''
#     def genere_doc(self) -> str:
#         return self.__repr__()
# def __new__(cls):
#     return float.__new__(cls)

# def __init__(self, value):
#     float.__init__(self, value)

# def __float__(self):
#     return float.__float__(self)


# class Distance(float):
#     def __new__(cls, value):
#         print('New')
#         return super().__new__(cls, value)
#
#     def __init__(self, value):
#         print('Init')
#         float.__init__(self)


class Distance(Measure):
    units = 'm'
    # def __float__(self):
    #     return float.__float__(self)

