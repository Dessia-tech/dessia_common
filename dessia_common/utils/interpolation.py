#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation tools

"""

from typing import List


def istep_from_value_on_list(list_: List[float], value: float,
                             extrapolate=False):
    """
    Return the float index of a value in a list of objects
    """
    for ipoint, (point1, point2) in enumerate(zip(list_[:-1],
                                                  list_[1:])):

        point1_s, point2_s = sorted((point1, point2))
        if point1_s <= value <= point2_s:
            alpha = (value - point1_s) / (point2_s - point1_s)
            if alpha < 0 or alpha > 1:
                raise ValueError
            return ipoint + alpha
    # values = [p for p in list_]

    if extrapolate:
        if abs(list_[0] - value) < abs(list_[-1] - value):
            # Closer to left
            return 0.
        # else:
            # Closer to right
        return len(list_) - 1

    min_values = min(list_)
    max_values = max(list_)
    raise ValueError(f'Specified value not found in list_: {value} not in [{min_values}, {max_values}]')


def interpolate_from_istep(objects, istep: float):
    """
    Return the interpolated object from a float index and a list of objects
    """
    n_objects = len(objects)
    if (istep < 0) or (istep > n_objects - 1):
        raise ValueError('Extrapolating is not supported')
    istep1 = int(istep)
    if istep1 == istep:
        # No interpolation needed
        return objects[int(istep)]
    # else:
    alpha = istep - istep1
    point1 = objects[istep1]
    point2 = objects[istep1 + 1]
    return (1 - alpha) * point1 + (alpha) * point2
