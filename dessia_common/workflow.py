#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

class Model:
    def __init__(self, object_class, object_id=None):
        self.object_class = object_class

class Function:
    def __init__(self, function):
        self.function = function
        self.input_args = []
        for arg in inspect.getargspec(function).args:
            
        

class Pipe:
    def __init__(self,
                 block1, variable_name1,
                 block2, variable_name2):
        self.block1 = block1
        self.block2 = block2
        self.variable_name1 = variable_name1
        self.variable_name2 = variable_name2

class AutomaticSort:
    def __init__(self):
        pass

class MileStone:
    def __init__(self):
        pass

class WorkFlow:
    def __init__(self, models, functions, pipes):
        self.models = models
        self.functions = functions
        self.pipes = pipes
        

class InputFunctionVariable:
    def __init__(self, function, arg_name):
        self.function = function
        self.arg_name = arg_name

        self.value = None

class OutputFunctionVariable:
    def __init__(self, function):
        self.function = function
        
        self.value = None


class ModelAttribute:
    def __init__(self, model, name):
        self.model = model
        self.name = name
    
        
import math

def sinus_f(x):
    return math.sin(x)

sinus = Function(sinus_f)

workflow = WorkFlow([], [sinus], [])