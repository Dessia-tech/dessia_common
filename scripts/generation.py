#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from dessia_common.generation import RegularDecisionTreeGenerator, DecisionTreeGenerator


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


class RegularCustomModelGenerator(RegularDecisionTreeGenerator):
    def __init__(self, name: str = ''):
        RegularDecisionTreeGenerator.__init__(self, number_possibilities=[2, 2, 2], name=name)

    def model_from_vector(self, vector):
        return Model(vector)

    def is_model_valid(self, model):
        return model.is_valid()


class CustomModelGenerator(DecisionTreeGenerator):
    def __init__(self, name: str = ''):
        DecisionTreeGenerator.__init__(self, name=name)

    def model_from_vector(self, vector):
        return Model(vector)

    def is_model_valid(self, model):
        return model.is_valid()

    def number_possibilities_from_model(self, model):
        if len(model.items) < 3:
            return 2
        return 0


generator = CustomModelGenerator()

models = []
for model in generator.generate(verbose=True):
    print(model)
    models.append(model)

assert len(models) == 4

regular_generator = RegularCustomModelGenerator()

regular_models = []
for model in regular_generator.generate(verbose=True):
    print(model)
    regular_models.append(model)

assert len(regular_models) == 4
