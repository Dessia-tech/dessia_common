import math
import dessia_common as dc
from dessia_common.utils.types import isinstance_base_types, is_sequence, full_classname
from copy import deepcopy as copy_deepcopy


def deepcopy(value, memo):

    if isinstance(value, type):  # For class
        return value

    elif isinstance(value, (float, int)):
        return value

    elif hasattr(value, '__deepcopy__'):

        memo_value = search_memo(value, memo)
        if memo_value is not None:
            return memo_value

        try:
            copied_value = value.__deepcopy__(memo)
        except TypeError:
            copied_value = value.__deepcopy__()
        memo[value] = copied_value
        print('1', value, '=>', copied_value)
        return copied_value

    else:
        if is_sequence(value):
            return deepcopy_sequence(value, memo)

        elif isinstance(value, dict):
            return deepcopy_dict(value, memo)

        else:
            new_value = copy_deepcopy(value, memo=memo)
            # print(value, "=>", new_value)
            memo[value] = new_value
            print('5', value, '=>', new_value)
            return new_value


def deepcopy_dict(dict_value, memo):
    memo_value = search_memo(dict_value, memo)
    if memo_value is not None:
        return memo_value

    copied_dict = {}
    for k, v in dict_value.items():
        copied_k = deepcopy(k, memo=memo)
        copied_v = deepcopy(v, memo=memo)
        try:
            memo[k] = copied_k
            print('3', k, '=>', copied_k)
        except TypeError:
            pass
        try:
            memo[v] = copied_v
            print('4', v, '=>', copied_v)
        except TypeError:
            pass
        copied_dict[copied_k] = copied_v
    return copied_dict


def deepcopy_sequence(seq_value, memo):
    memo_value = search_memo(seq_value, memo)
    if memo_value is not None:
        return memo_value

    copied_list = []
    for v in seq_value:
        cv = deepcopy(v, memo=memo)
        try:
            memo[v] = cv
            print('2', v, '=>', cv)
        except TypeError:
            pass
        copied_list.append(cv)
    return copied_list


def search_memo(value, memo):
    for key in memo.keys():
        if isinstance(value, type(key)) and value == key:
            print('0', value, '=>', (key, memo[key]))
            return memo[value]
    return None
