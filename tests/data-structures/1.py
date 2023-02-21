#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple data structure with non standalone objects & pointers


"""
from typing import List
import dessia_common.core as dc


class ShaftTest(dc.DessiaObject):
    _eq_is_data_eq = False

    def __init__(self, name=''):
        super().__init__(name=name)

    def __repr__(self):
        return f'Shaft {self.name}'


class ComponentTest(dc.DessiaObject):
    _eq_is_data_eq = False

    def __init__(self, shaft: ShaftTest, name=''):
        self.shaft = shaft
        super().__init__(name=name)


class ArchitectureTest(dc.DessiaObject):

    def __init__(self, list_shaft: List[ShaftTest], components: List[ComponentTest], name=''):
        self.list_shaft = list_shaft
        self.components = components
        super().__init__(name=name)


shaft_1 = ShaftTest('shaft 1')
shaft_2 = ShaftTest('shaft 2')
component_1 = ComponentTest(shaft=shaft_1)
component_2 = ComponentTest(shaft=shaft_2)

architecture = ArchitectureTest(list_shaft=[shaft_1, shaft_2], components=[component_1, component_2])
architecture.list_shaft.index(architecture.components[0].shaft)

dict_ = architecture.to_dict()
architecture_dict_to_object = ArchitectureTest.dict_to_object(dict_)

architecture_dict_to_object.list_shaft.index(architecture_dict_to_object.components[0].shaft)
