#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for errors in dessia_common.
"""

class ExceptionWithTraceback(Exception):
    """
    Base class with a message and a traceback.
    """

    def __init__(self, message, traceback_=''):
        super().__init__(message)
        self.message = message
        self.traceback = traceback_

    def __str__(self):
        return f'{self.message}\nTraceback:\n{self.traceback}'


class DeepAttributeError(ExceptionWithTraceback, AttributeError):
    """
    Abstract class for handling deep attribute errors.
    """


class ModelError(Exception):
    """
    Abstract class for handling model errors.
    """


class ConsistencyError(Exception):
    """
    Abstract class for handling consistency errors.
    """


class SerializationError(Exception):
    """
    Abstract class for handling serialization errors.
    """


class DeserializationError(Exception):
    """
    Abstract class for handling deserialization errors.
    """


class CopyError(Exception):
    """
    Abstract class for handling copy errors.
    """


class UntypedArgumentError(Exception):
    """
    Abstract class for handling untyped argument errors.
    """
