from dessia_common.core import DessiaObject
from typing import List


class ObjectTest(DessiaObject):
    _standalone_in_db = True
    def __init__(self, value_1: float, name= 'ObjectTest'):
        self.value_1 = value_1
        DessiaObject.__init__(self, name=name)

class AssemblyObjectTest(DessiaObject):
    _standalone_in_db = True
    def __init__(self, object_test: ObjectTest, value_2: float, name= 'ObjectTest'):
        self.object_test = object_test
        self.value_2 = value_2
        DessiaObject.__init__(self, name=name)


class ListAssemblyObjectTest(DessiaObject):
    _standalone_in_db = True
    def __init__(self, list_assembly: List[AssemblyObjectTest], name='ObjectTest'):
        self.list_assembly = list_assembly
        DessiaObject.__init__(self, name=name)
