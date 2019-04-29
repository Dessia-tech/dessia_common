#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

class Block:
    def __init__(self, object_class):
        self.object_class = object_class

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
    def __init__(self, blocks, pipes):
        self.blocks = blocks
        self.pipes = pipes