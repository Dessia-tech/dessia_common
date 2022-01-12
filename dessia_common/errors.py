#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


class ExceptionWithTraceback(Exception):
    def __init__(self, message, traceback_=''):
        self.message = message
        self.traceback = traceback_

    def __str__(self):
        return '{}\nTraceback:\n{}'.format(self.message, self.traceback)


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


class UntypedArgumentError(Exception):
    pass
