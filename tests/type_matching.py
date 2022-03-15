from dessia_common.utils.types import typematch
from dessia_common import DessiaObject
from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues
from typing import List, Tuple, Union, Any, get_args, get_origin, get_type_hints
from dessia_common.typings import Measure

assert typematch(DessiaObject, Any)
assert typematch(int, Any)

# DessiaObject should pass a test against object, but not the other way around
assert typematch(DessiaObject, object)  # DessiaObject IS an object
assert not typematch(object, DessiaObject)  # object COULD BE a DessiaObject but not necessarily => must raise an error

assert typematch(DessiaObject, Union[DessiaObject, int])

assert typematch(list[int], list[int])
assert typematch(List[Measure], List[float])
assert not typematch(List[float], List[Measure])
assert typematch(List[StandaloneObjectWithDefaultValues], List[DessiaObject])
assert typematch(List[StandaloneObjectWithDefaultValues], List[StandaloneObject])

assert typematch(tuple[int, str], tuple[int, str])
assert not typematch(tuple[int, int], tuple[str, int])



