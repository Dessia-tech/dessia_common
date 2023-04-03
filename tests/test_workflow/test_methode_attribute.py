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
        GMA = GetModelAttribute(AttributeType(DummyClassForTesting,name))
        if type_:
            TV = TypedVariable(type_=type_, name='Model attribute')
            self.assertTrue(GMA.outputs[0].name == TV.name)
            self.assertTrue(GMA.outputs[0].type_ == TV.type_)
        else:
            V = Variable(name='Model attribute')
            self.assertTrue(GMA.outputs[0]._data_eq(V))


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
        GMA = GetModelAttribute(AttributeType(DummyClassForTesting,name))
        self.assertTrue(GMA.equivalent(GMA.__deepcopy__()))


class TestSetModelAttribute(unittest.TestCase):          
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
        SMA = SetModelAttribute(AttributeType(DummyClassForTesting, name))
        TV_output = TypedVariable(type_=DummyClassForTesting, name='Model with changed attribute '+name)
        self.assertTrue(SMA.outputs[0].name == TV_output.name)
        self.assertTrue(SMA.outputs[0].type_ == TV_output.type_)
        if type_:
            TV_input = TypedVariable(type_=type_, name='Value to insert for attribute '+name)
            self.assertTrue(SMA.inputs[1].name == TV_input.name)
            self.assertTrue(SMA.inputs[1].type_ == TV_input.type_)
        else:
            V_input = Variable(name='Value to insert for attribute '+name)
            self.assertTrue(SMA.inputs[1]._data_eq(V_input))
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
        SMA = SetModelAttribute(AttributeType(DummyClassForTesting,name))
        self.assertTrue(SMA.equivalent(SMA.__deepcopy__()))

if __name__ == '__main__':
    unittest.main()
