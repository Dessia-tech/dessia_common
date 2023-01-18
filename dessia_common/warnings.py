#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that stores warnings in dessia_common.
"""


class SerializationWarning(Warning):
    """ Warning during serialization. """

    def __init__(self, message: str):
        self.message = message

        Warning.__init__(self, message)
