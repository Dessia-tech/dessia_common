from dessia_common.core import DessiaObject
from dessia_common.utils.diff import diff

value1 = {"a": 1, "b": 2, "c": 3}
value2 = {"a": 1, "b": 2}
value3 = {"a": 1}

# Checking that full path are given in result
assert diff(value1=value1, value2=value2).__repr__() == "Missing attributes:\t\t#/c: missing in second object\n"
assert diff(value1=value1, value2=value3).__repr__() == "Missing attributes:\t\t#/b: missing in second object\n\t" \
                                                        "#/c: missing in second object\n"

# Checking that diff is robust to commutation
assert diff(value1=value2, value2=value3).__repr__() == "Missing attributes:\t\t#/b: missing in second object\n"
assert diff(value1=value3, value2=value2).__repr__() == "Missing attributes:\t\t#/b: missing in first object\n"


class ObjTestEq(DessiaObject):
    """ Test Simple equalities. """
    _non_data_eq_attributes = ['arg3']
    _non_data_hash_attributes = ['arg3']

    def __init__(self, arg1: int, arg2: int):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = 0

        DessiaObject.__init__(self, name="")


obj_1 = ObjTestEq(1, 2)
obj_2 = ObjTestEq(1, 2)
assert obj_2 == obj_1
obj_1.arg3 = 5
assert obj_2 == obj_1

print("script diff.py has passed")
