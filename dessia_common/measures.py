#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from dessia_common.utils.types import get_python_class_from_class_name


# Measures
class Measure(float):
    si_unit = ''

    def to_dict(self):
        return {'object_class': f'{self.__module__}.{self.__class__.__name__}',
                'value': self.real}

    @classmethod
    def dict_to_object(cls, dict_):
        class_ = get_python_class_from_class_name(dict_['object_class'])
        return class_(dict_['value'])

    def __str__(self):
        return f'{self.__class__.__name__} {round(self, 6)} {self.si_unit}'


class Distance(Measure):
    si_unit = 'm'


class Angle(Measure):
    si_unit = 'rad'


class Torque(Measure):
    si_unit = 'Nm'


class Stress(Measure):
    si_unit = 'Pa'


class Time(Measure):
    si_unit = 's'


class Speed(Measure):
    si_unit = 'm/s'


class Acceleration(Measure):
    si_unit = 'm/s²'


class Mass(Measure):
    si_unit = 'kg'


class Force(Measure):
    si_unit = 'N'


class Work(Measure):
    si_unit = 'N*m'


class Power(Measure):
    si_unit = 'N*m/s'
