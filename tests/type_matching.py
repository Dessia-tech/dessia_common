from dessia_common.utils.types import typematch
from dessia_common import DessiaObject
from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues
from typing import List, Tuple, Union, Any, Optional
from dessia_common.typings import Measure

assert typematch(DessiaObject, Any)
assert typematch(int, Any)

# INHERITANCE
# DessiaObject should pass a test against object, but not the other way around
assert typematch(DessiaObject, object)  # DessiaObject IS an object
assert not typematch(object, DessiaObject)  # object COULD BE a DessiaObject but not necessarily => must raise an error

# LISTS
assert typematch(List[int], List[int])
assert typematch(List[Measure], List[float])
assert not typematch(List[float], List[Measure])
assert typematch(List[StandaloneObjectWithDefaultValues], List[DessiaObject])
assert typematch(List[StandaloneObjectWithDefaultValues], List[StandaloneObject])

# TUPLES
assert typematch(Tuple[int, str], Tuple[int, str])
assert not typematch(Tuple[int, int], Tuple[str, int])

# DEFAULT VALUES
assert typematch(Optional[List[StandaloneObject]], List[DessiaObject])
assert typematch(Union[List[StandaloneObject], type(None)], List[DessiaObject])

# UNION
assert typematch(DessiaObject, Union[DessiaObject, int])
assert not typematch(DessiaObject, Union[str, int])
assert not typematch(Union[DessiaObject, int], DessiaObject)
assert typematch(StandaloneObjectWithDefaultValues, Union[DessiaObject, StandaloneObject])
assert typematch(Union[str, int], Union[str, int])
# assert not typematch(Union[str, int], Union[str, int, bool])  # TODO Not implemented yet
assert typematch(Union[str, int], Union[int, str])

# UNEQUAL COMPLEX
assert not typematch(List[int], Tuple[int])

