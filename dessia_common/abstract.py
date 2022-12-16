"""
Abstract module that defines a base DessiaObject in order to avoid circular imports.
"""
from abc import ABC


class CoreDessiaObject(ABC):
    pass

    # @abstractmethod
    # def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#'):
    #     pass

    # @classmethod
    # @abstractmethod
    # def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
    #                     pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'CoreDessiaObject':
    #     pass

    # @abstractmethod
    # def copy(self, deep=True, memo=None):
    #     pass

    # @abstractmethod
    # def _serializable_dict(self):
    #     pass