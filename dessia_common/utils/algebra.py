#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algebra functions
"""
from functools import reduce


def number2factor(number):
    """
    Temporary function : Add to some tools package
    Finds all the ways to combine elements
    """
    factor_range = range(1, int(number ** 0.5) + 1)

    if number:
        factors = list(set(reduce(list.__add__, ([i, number // i]
                                                 for i in factor_range
                                                 if number % i == 0))))

        grids = [(factor_x, int(number / factor_x))
                 for factor_x in factors
                 if (number / factor_x).is_integer()]
    else:
        grids = []
    return grids


def number3factor(number, complete=True):
    """
    Temporary function : Add to some tools package
    Finds all the ways to combine elements
    """
    factor_range = range(1, int(number ** 0.5) + 1)

    if number:
        factors = list(set(reduce(list.__add__, ([i, number // i]
                                                 for i in factor_range
                                                 if number % i == 0))))
        if not complete:
            grids = get_incomplete_factors(number, factors)

        else:
            grids = [(factor_x, factor_y, int(number / (factor_x * factor_y)))
                     for factor_x in factors
                     for factor_y in factors
                     if (number / (factor_x * factor_y)).is_integer()]
        return grids
    return []


def get_incomplete_factors(number, factors):
    """
    TODO
    """
    grids = []
    sets = []
    for factor_x in factors:
        for factor_y in factors:
            value = number / (factor_x * factor_y)
            if value.is_integer():
                grid = (factor_x, factor_y, int(value))
                if set(grid) not in sets:
                    sets.append(set(grid))
                    grids.append(grid)
    return grids
