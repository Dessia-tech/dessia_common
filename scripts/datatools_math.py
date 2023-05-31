"""
Tests for dessia_common.datatools.math.
"""
from dessia_common.datatools.math import maximums, minimums

int_vector = [1, 2, 3, 4, 5, 6]

assert(maximums(int_vector*6) == [6])
assert(minimums(int_vector*6) == [1])
