#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:48:21 2023

@author: masfaraud
"""

"""

"""
from typing import List
import dessia_common.core as dc

import dessia_common.workflow as wf
import dessia_common.workflow.core as wf_core


class ShaftTest(dc.DessiaObject):
    _eq_is_data_eq = False

    def __init__(self):
        super().__init__(name='Shaft Test')


class ComponentTest(dc.DessiaObject):
    _eq_is_data_eq = False

    def __init__(self, shaft: ShaftTest):
        self.shaft = shaft
        super().__init__(name='Component Test')


class ArchitectureTest(dc.DessiaObject):

    def __init__(self, list_shaft: List[ShaftTest], components: List[ComponentTest]):
        self.list_shaft = list_shaft
        self.components = components
        super().__init__(name='Architecture Test')

    def __eq__(self, other_architecture):

        if len(self.list_shaft) == 2:
            return True
        return False

    def __hash__(self):
        return 1


class Model3dTest(dc.DessiaObject):
    def __init__(self, architecture: ArchitectureTest, bearing_shaft: ShaftTest):
        self.architecture = architecture
        self.bearing_shaft = bearing_shaft
        super().__init__(name='Model3d Test')

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#',
                id_method=True, id_memo=None):
        if memo:
            new_memo = {}
        else:
            new_memo = memo
        d = super().to_dict(use_pointers=use_pointers,
                            memo=new_memo,
                            path=path,
                            id_method=id_method,
                            id_memo=id_memo)
        for key, element in new_memo.items():
            memo[key] = element

        return d


class ListModel3dTest(dc.DessiaObject):

    def __init__(self, list_model3d: List[Model3dTest]):
        self.list_model3d = list_model3d
        super().__init__(name='List Model3d Test')


shaft_1 = ShaftTest()
shaft_2 = ShaftTest()
component_1 = ComponentTest(shaft=shaft_1)
component_2 = ComponentTest(shaft=shaft_2)

architecture = ArchitectureTest(list_shaft=[shaft_1, shaft_2], components=[component_1, component_2])
architecture.list_shaft.index(architecture.components[0].shaft)

dico_architecture = architecture.to_dict()
architecture_dict_to_object = ArchitectureTest.dict_to_object(dico_architecture)

architecture_dict_to_object.list_shaft.index(architecture_dict_to_object.components[0].shaft)

model3d = Model3dTest(architecture=architecture, bearing_shaft=shaft_1)
model3d_copy = model3d.copy(deep=True)

list_model3d_test = ListModel3dTest(list_model3d=[model3d, model3d_copy])

dico_list_model3d_test = list_model3d_test.to_dict()
list_model3d_test_dict_to_object = ListModel3dTest.dict_to_object(dico_list_model3d_test)

list_model3d_test_dict_to_object.list_model3d[0].architecture.list_shaft.index(
    list_model3d_test_dict_to_object.list_model3d[0].bearing_shaft)

list_model3d_test_dict_to_object.list_model3d[1].architecture.list_shaft.index(
    list_model3d_test_dict_to_object.list_model3d[1].bearing_shaft)

list_model_block = wf.InstantiateModel(ListModel3dTest, 'instanciate model list model 3d')

workflow = wf.Workflow(blocks=[list_model_block], output=list_model_block.outputs[0],
                       pipes=[])

input_values = {workflow.input_index(list_model_block.inputs[0]): [model3d]}

workflow_run = workflow.run(input_values=input_values)

workflow_run_dico = workflow_run.to_dict()

workflow_run_dict_to_object = wf_core.WorkflowRun.dict_to_object(workflow_run_dico)

list_model3d_workflow_test = workflow_run_dict_to_object.output_value

list_model3d_workflow_test.list_model3d[0].architecture.list_shaft.index(
    list_model3d_workflow_test.list_model3d[0].architecture.components[0].shaft)
