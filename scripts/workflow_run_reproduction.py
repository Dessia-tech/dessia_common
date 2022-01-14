#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:54:34 2019

@author: jezequel
"""

from dessia_common.workflow import Workflow, InstanciateModel,\
    ModelMethod, Pipe, WorkflowRun
from dessia_common.forms import Generator
from dessia_common.typings import MethodType

from dessia_api_client.users import PlatformUser

generator_block = InstanciateModel(model_class=Generator, name="Inst")
method_type = MethodType(class_=Generator, name="generate")
generate_block = ModelMethod(method_type=method_type, name="Meth")
display_attributes = ['intarg', 'strarg']

blocks = [generator_block, generate_block]
pipes = [
    Pipe(generator_block.outputs[0], generate_block.inputs[0])
]

workflow_ = Workflow(blocks=blocks, pipes=pipes,
                     output=generate_block.outputs[0])

workflow_run = workflow_.run({0: 10})

d = workflow_run.to_dict()
obj = WorkflowRun.dict_to_object(d)

workflow_run._check_platform()

u = PlatformUser(api_url="https://api.platform-dev.dessia.tech")
r = u.objects.create_object_from_python_object(workflow_run)
