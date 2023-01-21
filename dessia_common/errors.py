#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for errors in dessia_common.
"""


class ExceptionWithTraceback(Exception):
    """ Base class with a message and a traceback. """

    def __init__(self, message, traceback_=''):
        super().__init__(message)
        self.message = message
        self.traceback = traceback_

    def __str__(self):
        return f'{self.message}\nTraceback:\n{self.traceback}'


class DeepAttributeError(ExceptionWithTraceback, AttributeError):
    """ Error when trying to reach a deep attribute of an object, but couldn't get it. """


class ModelError(Exception):
    """ Error of model. """


class ConsistencyError(Exception):
    """ Error of consistency. """


class SerializationError(Exception):
    """ Error of serialization. """


class DeserializationError(Exception):
    """ Error of deserialization. """


class CopyError(Exception):
    """ Error of copy. """


class UntypedArgumentError(Exception):
    """ Error of code annotation. """
