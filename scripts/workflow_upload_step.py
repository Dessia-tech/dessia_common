#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:52:12 2022

@author: bricaud
"""

import dessia_common.workflow as wf
import dessia_common.tests as tests
from dessia_api_client import Client

block_component = wf.InstantiateModel(tests.Component, name = 'Compo')
upload_step_method = wf.MethodType(class_=tests.Component, name = 'upload_step')
method_upload_step = wf.ModelMethod(method_type = upload_step_method, name = 'upload_step')

workflow = wf.Workflow(blocks = [block_component, method_upload_step],
                       pipes = [wf.Pipe(block_component.outputs[0], method_upload_step.inputs[0])],
                       output = method_upload_step.outputs[0],
                       name = 'wf_upload_step')

dic_corresp = {i: j.name for i, j in enumerate(workflow.inputs)}

# workflow.plot()

c = Client(api_url='https://api.renault.dessia.tech')
r = c.create_object_from_python_object(workflow)