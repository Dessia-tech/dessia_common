#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" General checks & checklists. """

import time
import json
from typing import List
from dessia_common.abstract import CoreDessiaObject


LEVEL_TO_INT = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3}


class PassedCheck:
    """ Denote the result of a check that has no error. """

    level = 'info'

    def __init__(self, message: str):
        self.message = message

    def __repr__(self):
        return f'[{self.level.upper()}: {self.__class__.__name__}] - {self.message}'

    def __bool__(self):
        return True

    def to_dict(self):
        """ Write check as a dict. Used for frontend display. """
        return {"level": self.level, "message": self.message, "object_class": self.__class__.__name__}


class CheckWarning(PassedCheck):
    """ Denote a check warning. """

    level = 'warning'


class FailedCheck(PassedCheck):
    """ Denote a failed check. """

    level = 'error'

    def __bool__(self):
        return False


class BadType(FailedCheck):
    """ Denote a failed check due to a bad type. """


class GeometricInconsistance(FailedCheck):
    """ Denote a failed check due to a geometric inconsistency. """


class NonSerializable(FailedCheck):
    """ Used when a class or an instance is not serializable. """


class NonCopyable(FailedCheck):
    """ Used when an instance can not be copied. """


class NonDisplayable(FailedCheck):
    """ Used when an instance can not be displayed. """


class BadStructure(FailedCheck):
    """ Used when a class or method is badly designed (regarding schemas for example). """


class CheckList:
    """ A list of checks result. """

    def __init__(self, checks: List[PassedCheck]):
        self.checks = checks

    def __repr__(self):
        rep = f'Check list containing {len(self.checks)} checks:\n'
        for check_idx, check in enumerate(self.checks):
            rep += f'Check {check_idx+1} {check}\n'
        return rep

    def __add__(self, other_checklist: 'CheckList'):
        return self.__class__(self.checks + other_checklist.checks)

    def __len__(self):
        return len(self.checks)

    def __getitem__(self, item: int):
        return self.checks[item]

    def checks_above_level(self, level: str = 'error'):
        """ Return True if no check has a level above given one, else False. """
        checks = []
        for check in self.checks:
            if LEVEL_TO_INT[check.level] >= LEVEL_TO_INT[level]:
                checks.append(check)
        return checks

    def raise_if_above_level(self, level: str = 'error'):
        """ Raise an error if some checks have a level above given one. """
        for check in self.checks_above_level(level=level):
            raise ValueError(f'Check: {check} is above level "{level}"')

    def to_dict(self):
        """ Write CheckList as a dict. Used for frontend display. """
        return {"checks": [c.to_dict() for c in self.checks]}


def is_int(value, level: str = 'error'):
    """ Return if value is a int. """
    if not isinstance(value, int):
        return CheckList([FailedCheck(f'Value {value} is not an int')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is an int')])
    return CheckList([])


def is_float(value, level: str = 'error'):
    """ Return if value is a float. """
    if not isinstance(value, float):
        return CheckList([FailedCheck(f'Value {value} is not a float')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is a float')])
    return CheckList([])


def is_str(value, level: str = 'error'):
    """ Return if value is a str. """
    if not isinstance(value, str):
        return CheckList([FailedCheck(f'Value {value} is not a str')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is a str')])
    return CheckList([])


def type_check(value, expected_type, level: str = 'error'):
    """
    Type check the value against the expected type.

    This is experimental!
    """
    if expected_type is int:
        return is_int(value)
    if expected_type is float:
        return is_float(value)
    if expected_type is str:
        return is_str(value)

    if not isinstance(value, expected_type):
        return CheckList([FailedCheck(f'Value {value} is not of type {expected_type}')])
    if level == 'info':
        return CheckList([PassedCheck(f'value {value} is of expected type {expected_type}')])

    return CheckList([])


def check_serialization_process(object_: CoreDessiaObject, use_pointers: bool = True):
    """ Simulate platform serialization/deserialization process to guarantee viability of given object. """
    print("Checking serialization process...")
    start = time.time()
    try:
        dict_ = object_.to_dict(use_pointers=use_pointers)
        print("Serialized. Deserializing...")
    except TypeError as exc:
        if use_pointers:
            # Trying without pointers if it failed with.
            dict_ = object_.to_dict(use_pointers=False)
            print("Serialized without pointers. Deserializing...")
        else:
            print("Failed.\n")
            raise exc

    json_dict = json.dumps(dict_)
    decoded_json = json.loads(json_dict)
    deserialized_object = object_.dict_to_object(decoded_json)
    print("Deserialized. Checking equality...")

    if not deserialized_object._data_eq(object_):
        print('Failed.\nData Diff: ', object_._data_diff(deserialized_object))
        check = NonSerializable('Object is not equal to itself after serialization/deserialization')
    else:
        check = PassedCheck("Object is serializable")
    duration = time.time() - start
    if check:
        print(f"Checked serialization process in {duration}s.\n")
    return {"check": check, "duration": duration, "dict_": dict_}


def check_copy(object_: CoreDessiaObject):
    """ Simulate the copy performed by the platform. """
    print("Copying...")
    start = time.time()
    copied_object = object_.copy()
    print("Copied. Checking equality...")
    if not copied_object._data_eq(object_):
        print("Failed.\n")
        try:
            print('Data Diff: ', object_._data_diff(copied_object))
        except Exception:
            pass
        check = FailedCheck('Object is not equal to itself after copy.')
    else:
        check = PassedCheck("Object can be copied and equal to itself after copy.")
    duration = time.time() - start
    if check:
        print(f"Checked copying process in {duration}s.\n")
    return {"check": check, "duration": duration}


def check_displays(object_: CoreDessiaObject):
    """ Simulate displays computation performed by the platform. """
    print("Checking displays...")
    start = time.time()
    try:
        displays = object_._displays()
        print("Computed displays. Serializing displays...")
    except Exception:
        print("Failed.\n")
        return {"check": NonDisplayable("Object displays failed."), "duration": time.time() - start}

    try:
        json.dumps(displays)
        print("Serialized displays.")
    except Exception:
        print("Failed.\n")
        return {"check": NonSerializable("Object displays serialization failed."), "duration": time.time() - start}
    duration = time.time() - start
    print(f"Checked displays computation process in {duration}s.\n")
    return {"check": PassedCheck("Object displays checks succeeded"), "duration": duration}


def check_schemas(object_: CoreDessiaObject):
    """ Simulate schemas computation performed by the platform. """
    print("Checking schemas...")
    start = time.time()
    try:
        schemas = object_.method_schemas
        print("Computed schemas. Serializing schemas...")
    except Exception as exc:
        print("Failed.\n")
        return {"check": BadStructure(f"Schemas computation failed with following error : \n\n{exc}\n\n"),
                "duration": time.time() - start}

    try:
        json.dumps(schemas)
        print("Serialized schemas.")
    except Exception:
        print("Failed.\n")
        return {"check": NonSerializable("Schemas serialization failed."), "duration": time.time() - start}
    duration = time.time() - start
    print(f"Checked schemas computation process in {duration}s.\n")
    return {"check": PassedCheck("Object displays checks succeeded"), "duration": duration}
