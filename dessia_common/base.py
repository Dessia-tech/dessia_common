#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:39:07 2022

@author: steven
"""

import sys
from typing import Dict, Any
from dessia_common.typings import JsonSerializable
from dessia_common.utils.serialization import dict_to_object, serialize_dict_with_pointers, serialize_dict


class SerializableObject:
    """
    Serialization capabilities of Dessia Object
    """
    _non_serializable_attributes = []

    def base_dict(self):
        """
        A base dict for to_dict: put name, object class and version in a dict
        """
        package_name = self.__module__.split('.', maxsplit=1)[0]
        if package_name in sys.modules:
            package = sys.modules[package_name]
            if hasattr(package, '__version__'):
                package_version = package.__version__
            else:
                package_version = None
        else:
            package_version = None

        object_class = self.__module__ + '.' + self.__class__.__name__
        dict_ = {'object_class': object_class}
        if package_version:
            dict_['package_version'] = package_version
        return dict_

    def _serializable_dict(self):
        """
        Returns a dict of attribute_name, values (still python, not serialized)
        Keys are filtered with non serializable attributes controls
        """

        dict_ = {k: v for k, v in self.__dict__.items()
                 if k not in self._non_serializable_attributes and not k.startswith('_')}
        return dict_

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#') -> JsonSerializable:
        """
        Generic to_dict method
        """
        if memo is None:
            memo = {}

        # Default to dict
        serialized_dict = self.base_dict()
        dict_ = self._serializable_dict()
        if use_pointers:
            serialized_dict.update(serialize_dict_with_pointers(dict_, memo, path)[0])
        else:
            serialized_dict.update(serialize_dict(dict_))

        return serialized_dict

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: Dict[str, Any] = None, path: str = '#') -> 'SerializableObject':
        """
        Generic dict_to_object method
        """
        if 'object_class' in dict_:
            obj = dict_to_object(dict_=dict_, force_generic=force_generic, global_dict=global_dict,
                                 pointers_memo=pointers_memo, path=path)
            return obj

        if cls is not SerializableObject:
            obj = dict_to_object(dict_=dict_, class_=cls, force_generic=force_generic, global_dict=global_dict,
                                 pointers_memo=pointers_memo, path=path)
            return obj

        raise NotImplementedError('No object_class in dict')
