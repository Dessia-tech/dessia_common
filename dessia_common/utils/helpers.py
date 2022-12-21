"""
Tools for copying objects
"""

from typing import List

def concatenate(values):
    types_set = set(type(value) for value in values)
    concatenated_values = None
    if len(types_set) != 1:
        raise TypeError("Block Concatenate only defined for operands of the same type.")

    first_value = values[0]
    if hasattr(first_value, 'extend'):
        concatenated_values = first_value.__class__()
        for value in values:
            concatenated_values.extend(value)

    if concatenated_values is not None:
        return concatenated_values

    raise ValueError("Block Concatenate only defined for classes with 'extend' method")


def prettyname(name: str) -> str:
    """
    Creates a pretty name from as str
    """
    pretty_name = ''
    if name:
        strings = name.split('_')
        for i, string in enumerate(strings):
            if len(string) > 1:
                pretty_name += string[0].upper() + string[1:]
            else:
                pretty_name += string
            if i < len(strings) - 1:
                pretty_name += ' '
    return pretty_name

def maximums(matrix: List[List[float]]) -> List[float]:
    """
    Compute maximum values and store it in a list of length len(matrix[0]).
    """
    if not isinstance(matrix[0], list):
        return [max(matrix)]
    return [max(column) for column in zip(*matrix)]

def minimums(matrix: List[List[float]]) -> List[float]:
    """
    Compute minimum values and store it in a list of length len(matrix[0]).
    """
    if not isinstance(matrix[0], list):
        return [min(matrix)]
    return [min(column) for column in zip(*matrix)]
