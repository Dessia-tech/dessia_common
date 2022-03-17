"""
Tools for copying objects
"""

import warnings
import dessia_common as dc
import dessia_common.files
from dessia_common.utils.types import is_sequence, is_typing


def deepcopy_value(value, memo):

    if isinstance(value, type) or is_typing(value):  # For class
        return value

    elif isinstance(value, (float, int, str)):
        copied_value = value
        return copied_value

    elif value is None:
        return None

    elif value.__class__.__name__ in ['Point2D', 'Point3D',
                                      'Vector2D', 'Vector3D']:
        try:
            copied_value = value.copy(deep=True, memo=memo)
        except TypeError:
            warnings.warn(f'{value.__class__.__name__}.copy() does not implement deep and memo arguments')
            copied_value = value.copy()
        return copied_value

    elif isinstance(value, dc.DessiaObject):
        memo_value = search_memo(value, memo)
        if memo_value is not None:
            return memo_value
        try:
            copied_value = value.copy(deep=True, memo=memo)
        except TypeError:
            warnings.warn(f'{value.__class__.__name__}.copy() does not implement deep and memo arguments')
            copied_value = value.copy()

        memo[value] = copied_value
        return copied_value

    elif isinstance(value, (dessia_common.files.BinaryFile, dessia_common.files.StringFile)):
        return value.copy()

    elif hasattr(value, '__deepcopy__'):
        memo_value = search_memo(value, memo)
        if memo_value is not None:
            return memo_value

        try:
            copied_value = value.__deepcopy__(memo)
        except TypeError:
            copied_value = value.__deepcopy__()
        memo[value] = copied_value
        return copied_value

    else:
        if is_sequence(value):
            return deepcopy_sequence(value, memo)

        elif isinstance(value, dict):
            return deepcopy_dict(value, memo)

        else:
            raise NotImplementedError(f'unhandle type for copy: {value} of type {value.__class__}')


def deepcopy_dict(dict_value, memo):
    memo_value = search_memo(dict_value, memo)
    if memo_value is not None:
        return memo_value

    copied_dict = {}
    for k, v in dict_value.items():
        copied_k = deepcopy_value(k, memo=memo)
        copied_v = deepcopy_value(v, memo=memo)
        copied_dict[copied_k] = copied_v
    return copied_dict


def deepcopy_sequence(seq_value, memo):
    memo_value = search_memo(seq_value, memo)
    if memo_value is not None:
        return memo_value

    copied_list = []
    for v in seq_value:
        cv = deepcopy_value(v, memo=memo)
        copied_list.append(cv)
    return copied_list


def search_memo(value, memo):
    for key in memo.keys():
        if isinstance(value, type(key)) and value == key:
            return memo[value]
    return None
