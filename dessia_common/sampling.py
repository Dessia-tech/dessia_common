"""
Library for sampling data.

"""
from typing import List, Dict, Any, Type

import numpy as npy
import pyDOE2 as pyDOE
import matplotlib.pyplot as plt

from dessia_common.core import DessiaObject
from dessia_common.datatools import HeterogeneousList
from dessia_common.optimization import FixedAttributeValue, BoundedAttributeValue

class Sampler(DessiaObject):
    _standalone_in_db = True
    _vector_features = []

    def __init__(self, object_class: Type, sampled_attributes: List[BoundedAttributeValue],
                 constant_attributes: List[FixedAttributeValue], name: str = ''):
        self.object_class = object_class
        self.sampled_attributes = sampled_attributes
        self.constant_attributes = constant_attributes
        self.attributes = self._get_attributes_names()
        DessiaObject.__init__(self, name=name)

    def _get_attributes_names(self):
        return [attr.attribute_name for attr in self.constant_attributes + self.sampled_attributes]

    def _build_parameter_grid(self, instances_numbers: List[int]):
        parameter_grid = []
        for attr, instances_number in zip(self.constant_attributes, instances_numbers[:len(self.constant_attributes)]):
            parameter_grid.append([attr.value])

        for attr, instances_number in zip(self.sampled_attributes, instances_numbers[len(self.constant_attributes):]):
            parameter_grid.append(npy.linspace(attr.min_value, attr.max_value, instances_number, dtype=float).tolist())

        return parameter_grid

    def _get_doe(self):
        return

    def _lhs_sampling(self):
        return

    def _montecarlo_sampling(self):
        return

    def _full_factorial_sampling(self):
        instances_numbers = [1] * len(self.constant_attributes) + [4] * len(self.sampled_attributes)
        parameter_grid = self._build_parameter_grid(instances_numbers)
        idx_sampling = pyDOE.fullfact(instances_numbers)
        full_doe = []
        for idx_sample in idx_sampling:
            valued_sample = [parameter_grid[attr_row][int(idx)] for attr_row, idx in enumerate(idx_sample)]
            full_doe.append(self.object_class(**dict(zip(self.attributes, valued_sample))))
        return full_doe

