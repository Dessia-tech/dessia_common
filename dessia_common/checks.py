#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General checks & checklists
"""

from dessia_common.core import SerializableObject


class PassedCheck(SerializableObject):
    level = 'info'

    def __init__(self, message: str):
        self.message = message

    def __repr__(self):
        return f'[{self.level}]: {self.__class__.__name__} {self.message}'


class CheckWarning(PassedCheck):
    level = 'warning'


class FailedCheck(PassedCheck):
    level = 'error'


class BadType(FailedCheck):
    pass


class GeometricInconsistance(FailedCheck):
    pass


class CheckList(SerializableObject):
    def __init__(self, checks):
        self.checks = checks

    def __repr__(self):
        rep = ''
        for check in self.checks:
            rep += str(check)
        return rep

    def __add__(self, other_checklist):
        return self.__class__(self.checks + other_checklist.checks)


def is_int(value, level='error'):
    if not isinstance(value, int):
        return CheckList([FailedCheck(f'Value {value} is not an int')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is an int')])
    return CheckList([])


def is_float(value, level='error'):
    if not isinstance(value, float):
        return CheckList([FailedCheck(f'Value {value} is not an float')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is an float')])
    return CheckList([])


def is_str(value, level='error'):
    if not isinstance(value, int):
        return CheckList([FailedCheck(f'Value {value} is not an str')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is an str')])
    return CheckList([])


def type_check(value, expected_type, level='error'):
    if expected_type is int:
        return is_int(value)
    if expected_type is float:
        return is_float(value)
    if expected_type is str:
        return is_str(value)

    if not issubclass(value.__class__, expected_type):
        return CheckList([FailedCheck(f'Value {value} is not of type {expected_type}')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is of expected type {expected_type}')])

    return CheckList([])
