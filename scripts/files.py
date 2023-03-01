#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:10:54 2022

@author: steven
"""

import io
import importlib
import inspect

module = importlib.import_module('dessia_common.files')

for name, obj in inspect.getmembers(module):
    if inspect.isclass(obj) and issubclass(obj, (io.BytesIO, io.StringIO)):
        template = obj.save_template_to_file('test')
