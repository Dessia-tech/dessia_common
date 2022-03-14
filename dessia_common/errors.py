#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


class ExceptionWithTraceback(Exception):
    """
    Base class with a message and a traceback
    """

    def __init__(self, message, traceback_=''):
        super().__init__(message)
        self.message = message
        self.traceback = traceback_

    def __str__(self):
        return f'{self.message}\nTraceback:\n{self.traceback}'


class DeepAttributeError(ExceptionWithTraceback, AttributeError):
    pass


class ModelError(Exception):
    pass


class ConsistencyError(Exception):
    pass


class SerializationError(Exception):
    pass


class DeserializationError(Exception):
    pass


class CopyError(Exception):
    pass


class UntypedArgumentError(Exception):
    pass
