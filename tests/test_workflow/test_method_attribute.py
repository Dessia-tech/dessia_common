import unittest
from dessia_common.workflow.blocks import GetModelAttribute, SetModelAttribute
from dessia_common.workflow.core import Variable
from dessia_common.typings import AttributeType
from dessia_common.forms import DessiaObject
from typing import List, Dict
from parameterized import parameterized


class DummyClassForTesting(DessiaObject):
    def __init__(self, int_: int, variable_, float_: float = 0.0, str_: str = "test", bool_: bool = True,
                 list_: List[int] = None, dict_: Dict[str, int] = None, two: int = 2, name: str = "", **kwargs):
        self.int_ = int_
        variable_ = variable_
        self.float_ = float_
        self.str_ = str_
        self.bool_ = bool_
        if list_ is None:
            list_ = [1, 2, 3]
        self.list_ = list_
        if dict_ is None:
            dict_ = {"a": 1, "b": 2}
        self.dict_ = dict_
        self.two = two
        self.name = name
        self.combine = self.int_ * 5
        DessiaObject.__init__(self, **kwargs)
    
    def add_some_value(self) -> None:
        self.combine = self.int_ + self.two


class TestGetModelAttribute(unittest.TestCase):

    @parameterized.expand([
        (int, "int_"),
        (float, "float_"),
        (str, "str_"),
        (bool, "bool_"),
        (List[int], "list_"),
        (Dict[str, int], "dict_"),
        (int, "two"),
        (str, "name"),
        (None, "variable_"),
        (None, "combine"),
        (None, "kwargs")
    ])
    def test_outputs(self, type_, name):
        block = GetModelAttribute(AttributeType(DummyClassForTesting, name))
        output = block.outputs[0]
        variable = Variable(type_=type_, name="Model")
        self.assertEqual(output.type_, variable.type_)
        self.assertEqual(output.name, variable.name)

    @parameterized.expand([
        "int_",
        "float_",
        "str_",
        "bool_",
        "list_",
        "dict_",
        "two",
        "name",
        "variable_",
        "combine",
        "kwargs",
        "tuple_",
        "list_of_tuple"
    ])
    def test_getmethod_equivalent(self, name):
        block = GetModelAttribute(AttributeType(DummyClassForTesting, name))
        self.assertTrue(block.equivalent(block.__deepcopy__()))


class TestSetModelAttribute(unittest.TestCase):

    @parameterized.expand([
        (int, "int_"),
        (float, "float_"),
        (str, "str_"),
        (bool, "bool_"),
        (List[int], "list_"),
        (Dict[str, int], "dict_"),
        (int, "two"),
        (str, "name"),
        (None, "variable_"),
        (None, "combine"),
        (None, "kwargs")
    ])
    def test_outputs(self, type_, name):
        block = SetModelAttribute(AttributeType(DummyClassForTesting, name))
        output_variable = Variable(type_=DummyClassForTesting, name="Model")
        output = block.outputs[0]
        value_input = block.inputs[1]
        self.assertEqual(output.type_, output_variable.type_)
        value_variable = Variable(type_=type_, name="Value")
        self.assertEqual(value_input.type_, value_variable.type_)
        self.assertEqual(value_input.name, value_variable.name)

    @parameterized.expand([
        "int_",
        "float_",
        "str_",
        "bool_",
        "list_",
        "dict_",
        "two",
        "name",
        "variable_",
        "combine",
        "kwargs",
        "tuple_",
        "list_of_tuple"
    ])
    def test_setmethod_equivalent(self, name):
        block = SetModelAttribute(AttributeType(DummyClassForTesting, name))
        self.assertTrue(block.equivalent(block.__deepcopy__()))


if __name__ == "__main__":
    unittest.main()
