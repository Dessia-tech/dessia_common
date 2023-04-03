from dessia_common import tests
import unittest
from dessia_common import core
from dessia_common.workflow.blocks import GetModelAttribute, SetModelAttribute
from dessia_common.workflow.core import Variable, TypedVariable
import inspect
from dessia_common.typings import AttributeType
from dessia_common.forms import DessiaObject
import time
from typing import List, Dict

class UnrealClassForTesting(DessiaObject):
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

    def test_outputs(self):
        instanciate = [(GetModelAttribute(AttributeType(UnrealClassForTesting,'int_')), 
                        TypedVariable(type_=int, name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'float_')), 
                         TypedVariable(type_=float, name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'str_')), 
                         TypedVariable(type_=str, name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'bool_')), 
                         TypedVariable(type_=bool, name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'l')), 
                         TypedVariable(type_=List[int], name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'d')), 
                         TypedVariable(type_=Dict[str,int], name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'two')), 
                         TypedVariable(type_=int, name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'name')), 
                         TypedVariable(type_=str, name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'combine')), 
                         Variable(name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'variable_')), 
                         Variable(name='Model attribute')),
                        (GetModelAttribute(AttributeType(UnrealClassForTesting,'kwargs')), 
                         Variable(name='Model attribute'))]
        for el in instanciate:
            if type(el[1])==Variable:
                self.assertTrue(el[0].outputs[0]._data_eq(el[1]))
            else:
                self.assertTrue(el[0].outputs[0].name == el[1].name)
                self.assertTrue(el[0].outputs[0].type_ == el[1].type_)
        pass

    def test_getmethod_equivalent(self):
        instanciate = [GetModelAttribute(AttributeType(UnrealClassForTesting,'int_')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'float_')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'str_')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'bool_')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'l')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'d')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'two')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'name')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'tuple_')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'list_of_tuple')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'combine')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'variable_')),
                        GetModelAttribute(AttributeType(UnrealClassForTesting,'kwargs'))]
        for el in instanciate:
            self.assertTrue(el.equivalent(el.__deepcopy__()))


class TestSetModelAttribute(unittest.TestCase):          
    def setUp(self):
        pass

    def test_outputs(self):
        instanciate = [(SetModelAttribute(AttributeType(UnrealClassForTesting,'int_')),
                        TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute int_'), 
                        TypedVariable(type_=int, name='Value to insert for attribute int_')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'float_')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute float_'), 
                         TypedVariable(type_=float, name='Value to insert for attribute float_')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'str_')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute str_'), 
                         TypedVariable(type_=str, name='Value to insert for attribute str_')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'bool_')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute bool_'), 
                         TypedVariable(type_=bool, name='Value to insert for attribute bool_')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'l')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute l'), 
                         TypedVariable(type_=List[int], name='Value to insert for attribute l')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'d')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute d'), 
                         TypedVariable(type_=Dict[str,int], name='Value to insert for attribute d')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'two')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute two'), 
                         TypedVariable(type_=int, name='Value to insert for attribute two')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'combine')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute combine'), 
                         Variable(name='Value to insert for attribute combine')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'variable_')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute variable_'), 
                         Variable(name='Value to insert for attribute variable_')),
                        (SetModelAttribute(AttributeType(UnrealClassForTesting,'kwargs')), 
                         TypedVariable(type_=UnrealClassForTesting, name='Model with changed attribute kwargs'), 
                         Variable(name='Value to insert for attribute kwargs'))]
        for el in instanciate:
            self.assertTrue(el[0].outputs[0].name == el[1].name)
            self.assertTrue(el[0].outputs[0].type_ == el[1].type_)
            if type(el[2])==Variable:
                self.assertTrue(el[0].inputs[1]._data_eq(el[2]))
            else:
                self.assertTrue(el[0].inputs[1].name == el[2].name)
                self.assertTrue(el[0].inputs[1].type_ == el[2].type_)
        pass

    def test_setmethod_equivalent(self):
        instanciate = [SetModelAttribute(AttributeType(UnrealClassForTesting,'int_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'float_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'str_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'bool_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'list_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'d')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'two')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'name')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'tuple_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'list_of_tuple')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'combine')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'variable_')),
                        SetModelAttribute(AttributeType(UnrealClassForTesting,'kwargs'))]
        for el in instanciate:
            self.assertTrue(el.equivalent(el.__deepcopy__()))

if __name__ == '__main__':
    unittest.main()
