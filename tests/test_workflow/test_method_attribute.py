import unittest
from dessia_common.workflow.blocks import GetModelAttribute, SetModelAttribute
from dessia_common.workflow.core import Variable, TypedVariable
from dessia_common.typings import AttributeType
from dessia_common.forms import DessiaObject
from typing import List, Dict
from parameterized import parameterized

class DummyClassForTesting(DessiaObject):
    def __init__(self, int_: int , variable_, float_: float = 0.0, str_: str = 'test', bool_: bool = True,
                  l: List[int] = [1,2,3], d: Dict[str,int] = {'a':1,'b':2}, two: int = 2, name: str ='', **kwargs):
        self.int_ = int_
        variable_= variable_
        self.float_ = float_
        self.str_ = str_
        self.bool_ = bool_
        self.list_ = l
        self.d = d
        self.two = two
        self.name = name
        self.combine = self.int_*5
        DessiaObject.__init__(self, **kwargs)
    
    def add_some_value(self) -> None:
        self.combine = self.int_+self.two
        return 


class TestGetModelAttribute(unittest.TestCase):          
    def setUp(self):
        pass

    @parameterized.expand([
        (int, 'int_'),
        (float, 'float_'),
        (str, 'str_'),
        (bool, 'bool_'),
        (List[int], 'l'),
        (Dict[str,int], 'd'),
        (int, 'two'),
        (str, 'name'),
        (None, 'variable_'),
        (None, 'combine'),
        (None, 'kwargs')])
    def test_outputs(self, type_, name):
        block = GetModelAttribute(AttributeType(DummyClassForTesting,name))
        if type_:
            typed_varible = TypedVariable(type_=type_, name='Model attribute')
            self.assertTrue(block.outputs[0].name == typed_varible.name)
            self.assertTrue(block.outputs[0].type_ == typed_varible.type_)
        else:
            variable = Variable(name='Model attribute')
            self.assertTrue(block.outputs[0]._data_eq(variable))

    @parameterized.expand([
        'int_',
        'float_',
        'str_',
        'bool_',
        'l',
        'd',
        'two',
        'name',
        'variable_',
        'combine',
        'kwargs',
        'tuple_',
        'list_of_tuple'])
    def test_getmethod_equivalent(self, name):
        block = GetModelAttribute(AttributeType(DummyClassForTesting,name))
        self.assertTrue(block.equivalent(block.__deepcopy__()))


class TestSetModelAttribute(unittest.TestCase):          
    def setUp(self):
        pass

    @parameterized.expand([
        (int, 'int_'),
        (float, 'float_'),
        (str, 'str_'),
        (bool, 'bool_'),
        (List[int], 'l'),
        (Dict[str, int], 'd'),
        (int, 'two'),
        (str, 'name'),
        (None, 'variable_'),
        (None, 'combine'),
        (None, 'kwargs')])
    def test_outputs(self, type_, name):
        block = SetModelAttribute(AttributeType(DummyClassForTesting, name))
        typed_variable_output = TypedVariable(type_=DummyClassForTesting, name="Model")
        self.assertEqual(block.outputs[0].type_, typed_variable_output.type_)
        if type_:
            typed_variable_input = TypedVariable(type_=type_, name="Value")
            self.assertEqual(block.inputs[1].type_, typed_variable_input.type_)
        else:
            variable_input = Variable(name='Value to insert for attribute '+name)
            self.assertTrue(block.inputs[1]._data_eq(variable_input))
        pass

    @parameterized.expand([
        'int_',
        'float_',
        'str_',
        'bool_',
        'l',
        'd',
        'two',
        'name',
        'variable_',
        'combine',
        'kwargs',
        'tuple_',
        'list_of_tuple'])
    def test_setmethod_equivalent(self, name):
        block = SetModelAttribute(AttributeType(DummyClassForTesting,name))
        self.assertTrue(block.equivalent(block.__deepcopy__()))


if __name__ == '__main__':
    unittest.main()
