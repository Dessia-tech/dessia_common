#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 18:59:29 2022

@author: masfaraud
"""

import inspect
import dessia_common.workflow.blocks as dcb

# TODO Build a real test.
for name, member in inspect.getmembers(dcb):
    if inspect.isclass(member) and issubclass(member, dcb.Block):
        print('testing block', name)
        member.schema()
