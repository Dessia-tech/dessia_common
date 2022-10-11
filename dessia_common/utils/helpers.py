"""
Tools for copying objects
"""

def concatenate(values):
    types_set = set(type(value) for value in values)
    concatenated_values = None
    if len(types_set) != 1:
        raise TypeError("Block Concatenate only defined for operands of the same type.")

    first_value = values[0]
    if hasattr(first_value, 'extend'):
        concatenated_values = values[0].__class__()
        for value in values:
            concatenated_values.extend(value)

    if concatenated_values is not None:
        return concatenated_values

    raise ValueError("Block Concatenate only defined for classes with 'extend' method")
