#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines physic quantities called measures.

It is an overloading of float, and usefull in platform forms to have access to units conversion.
"""

from dessia_common.utils.types import get_python_class_from_class_name


# Measures
class Measure(float):
    """Abstract base class of measures, do not instantiate directly."""
    si_unit = ''

    def to_dict(self):
        """
        Serializes the object into a dict.
        """
        return {'object_class': f'{self.__module__}.{self.__class__.__name__}',
                'value': self.real}

    @classmethod
    def dict_to_object(cls, dict_):
        """
        Unserializes the dict into an object.
        """
        class_ = get_python_class_from_class_name(dict_['object_class'])
        return class_(dict_['value'])

    def __str__(self):
        return f'{self.__class__.__name__} {round(self, 6)} {self.si_unit}'


class Distance(Measure):
    """
    Represent a distance in meters.
    """
    si_unit = 'm'


class Angle(Measure):
    """
    Represent an angle in radians.
    """
    si_unit = 'rad'


class Torque(Measure):
    """
    Represent a torque in newton-meter.
    """
    si_unit = 'Nm'


class Stress(Measure):
    """
    Represent a stress in pascal.
    """
    si_unit = 'Pa'


class Time(Measure):
    """
    Represent time in seconds.
    """
    si_unit = 's'


class Speed(Measure):
    """
    Represent a speed in meters per second.
    """
    si_unit = 'm/s'


class Acceleration(Measure):
    """
    Represent an acceleration in meter per second square.
    """
    si_unit = 'm/s²'


class Mass(Measure):
    """
    Represent a mass in kilogram.
    """
    si_unit = 'kg'


class Force(Measure):
    """
    Represent a force in newton.
    """
    si_unit = 'N'


class Work(Measure):
    """
    Represent a work in joules.
    """
    si_unit = 'N*m'


class Power(Measure):
    """
    Represent a power in watts.
    """
    si_unit = 'N*m/s'
