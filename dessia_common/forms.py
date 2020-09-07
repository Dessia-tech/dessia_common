"""
This is a module that aims to list all
possibilities of data formats offered by Dessia.
It should be used as a repertory of rules and available typings.

Some general rules :

- Lists are homogeneous, ie, they should not
  contain several types of elements
    ex : List[int], List[str], List[CustomClass], ...
- Tuples can, therefore, be used as heterogenous sequences,
  thanks to the fact that they are immutables.
    ex : Tuple[str, int, CustomClass] is a tuple like this :
        t = ('tuple', 1, custom_class_object)
- Dict is used whenever a dynamic structure should be defined.
  It takes only one possible type for keys
  and one possible type for values.
    ex : Dict[str, bool] is a dict like :
        d = {'key0': True, 'key1': False, 'another_key': False,...}
- As opposed to this, TypedDict defines a static structure,
  with a defined number of given, expected keys & types of their values.
    ex :
    class StaticDict(TypedDict):
        name: str
        value: float

In addition to types & genericity (brought by DessiaObject),
this module can also be seen as a template for Dessia's
coding/naming style & convention.
"""

from dessia_common import DessiaObject
from typing import Dict, List, Tuple, Union
try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


class StandaloneSubobject(DessiaObject):
    _standalone_in_db = True
    _generic_eq = True

    def __init__(self, floatarg: float, name: str = 'Standalone Subobject'):
        self.floatarg = floatarg

        DessiaObject.__init__(self, name=name)


class EnhancedStandaloneSubobject(StandaloneSubobject):
    def __init__(self, floatarg: float, boolarg: bool,
                 name: str = 'Standalone Subobject'):
        self.boolarg = boolarg

        StandaloneSubobject.__init__(self, floatarg=floatarg, name=name)


class EmbeddedSubobject(DessiaObject):
    def __init__(self, embedded_list: List[int] = None,
                 name: str = 'Embedded Subobject'):
        if embedded_list is None:
            self.embedded_list = [1, 2, 3]
        else:
            self.embedded_list = embedded_list

        DessiaObject.__init__(self, name=name)


class StaticDict(TypedDict):
    name: str
    value: float
    is_valid: bool
    subobject: EmbeddedSubobject


class StandaloneObject(DessiaObject):
    _standalone_in_db = True
    _generic_eq = True

    def __init__(self, standalone_subobject: StandaloneSubobject,
                 embedded_subobject: EmbeddedSubobject,
                 dynamic_dict: Dict[str, bool], static_dict: StaticDict,
                 tuple_arg: Tuple[str, int], intarg: int, strarg: str,
                 object_list: List[StandaloneSubobject],
                 subobject_list: List[EmbeddedSubobject],
                 builtin_list: List[int],
                 union_arg: Union[StandaloneSubobject,
                                  EnhancedStandaloneSubobject],
                 name: str = 'Standalone Object Demo'):
        self.union_arg = union_arg
        self.builtin_list = builtin_list
        self.subobject_list = subobject_list
        self.object_list = object_list
        self.tuple_arg = tuple_arg
        self.strarg = strarg
        self.intarg = intarg
        self.static_dict = static_dict
        self.dynamic_dict = dynamic_dict
        self.standalone_subobject = standalone_subobject
        self.embedded_subobject = embedded_subobject

        DessiaObject.__init__(self, name=name)

    def add_standalone_object(self, object_: StandaloneSubobject):
        """
        This methods adds a standalone object to object_list.
        It doesn't return anything, hence, API will update object when
        computing from frontend
        """
        self.object_list.append(object_)

    def add_embedded_object(self, object_: EmbeddedSubobject):
        """
        This methods adds an embedded object to subobject_list.
        It doesn't return anything, hence, API will update object
        when computing from frontend
        """
        self.subobject_list.append(object_)

    def add_float(self, value) -> StandaloneSubobject:
        """
        This methods adds value to its standalone subobject
        floatarg property and returns it.
        API should replace standalone_subobject as it is returned
        """
        self.standalone_subobject.floatarg += value
        return self.standalone_subobject
