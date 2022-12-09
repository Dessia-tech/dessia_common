"""
Abstract module that defines a base DessiaObject in order to avoid circular imports
"""
from abc import ABC, abstractmethod


class CoreDessiaObject(ABC):

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def dict_to_object(self, dict_):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def _serializable_dict(self):
        pass
