"""
Tools for copying objects
"""

import dessia_common as dc

def concatenate(*values):
    types_set = set(type(value) for value in values)
    print('HEEEEERRRRRREEEEEE', types_set)
    if len(types_set) != 1:
        raise TypeError("Block Concatenate only defined for operands of the same type.")

    first_value = values[0]
    if isinstance(first_value, list):
        concatenated_values = []
        for value in values:
            concatenated_values.extend(value)

    if isinstance(first_value, dict): # TODO manage same key behavior, maybe dict is not a good use case
        concatenated_values = values[0]
        for value in values[1:]:
            concatenated_values = dict(concatenated_values, **value)

    if isinstance(first_value, dc.HeterogeneousList): # TODO merge with list case when extend is developed in HList
        dessia_objects = []
        name = ''
        for value in values:
            dessia_objects.extend(value.dessia_objects)
            name += value.name + ("_" if value.name != "" else "")
        concatenated_values = dc.HeterogeneousList(dessia_objects, name)
    return concatenated_values
