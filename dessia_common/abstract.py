"""
Abstract module that defines a base DessiaObject in order to avoid circular imports
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dessia_common.typings import JsonSerializable


class CoreDessiaObject(ABC):

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'CoreDessiaObject':
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def _serializable_dict(self):
        pass
