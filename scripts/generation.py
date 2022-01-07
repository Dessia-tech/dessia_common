#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from dessia_common.generation import RegularDecisionTreeGenerator

class Model:
    def __init__(self, items):
        self.items = items

    def __repr__(self):
        return 'Model of items: {}'.format(self.items)        

    def is_valid(self):
        if len(self.items) >= 2:
            if self.items[1] == 0:
                return False
        return True
        


class CustomModelGenerator(RegularDecisionTreeGenerator):
    def __init__(self):
        RegularDecisionTreeGenerator.__init__(self, number_possibilities=[2,2,2])

    def model_from_vector(self, vector):
        return Model(vector)

    def is_model_valid(self, model):
        return model.is_valid()
        
generator = CustomModelGenerator()

for model in generator.generate():
    print(model)