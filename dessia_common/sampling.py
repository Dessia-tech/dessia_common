"""
Library for sampling data.

"""
from typing import List, Type

import random
import numpy as npy
import pyDOE2 as pyDOE

from dessia_common.core import DessiaObject
from dessia_common.datatools import HeterogeneousList
from dessia_common.optimization import FixedAttributeValue, BoundedAttributeValue

class Sampler(DessiaObject):
    """
    Base object to build a DOE from a class and choosen limits for all specified sampled_class attributes.

    :param sampled_class: Class type to sample
    :type sampled_class: `type`

    :param sampled_attributes: List of varying attributes in the DOE
    :type sampled_attributes: `List[BoundedAttributeValue]`

    :param constant_attributes: List of fixed attributes in the DOE
    :type constant_attributes: `List[FixedAttributeValue]`

    :param name: Name of Sampler
    :type name: `str`, `optional`, defaults to `''`

    """
    _standalone_in_db = True
    _vector_features = []

    def __init__(self, sampled_class: Type, sampled_attributes: List[BoundedAttributeValue],
                 constant_attributes: List[FixedAttributeValue], name: str = ''):
        self.sampled_class = sampled_class
        self.sampled_attributes = sampled_attributes
        self.constant_attributes = constant_attributes
        self.attributes = self._get_attributes_names()
        DessiaObject.__init__(self, name=name)

    def _get_attributes_names(self):
        return [attr.attribute_name for attr in self.constant_attributes + self.sampled_attributes]

    def _get_instances_numbers(self):
        return [1] * len(self.constant_attributes) + [attr.number for attr in self.sampled_attributes]

    def _build_parameter_grid(self, instances_numbers: List[int]):
        parameter_grid = []
        for attr, instances_number in zip(self.constant_attributes, instances_numbers[:len(self.constant_attributes)]):
            parameter_grid.append([attr.value])

        for attr, instances_number in zip(self.sampled_attributes, instances_numbers[len(self.constant_attributes):]):
            parameter_grid.append(npy.linspace(attr.min_value, attr.max_value, instances_number, dtype=float).tolist())

        return parameter_grid

    def _full_factorial_sampling(self):
        instances_numbers = self._get_instances_numbers()
        parameter_grid = self._build_parameter_grid(instances_numbers)
        idx_sampling = pyDOE.fullfact(instances_numbers)
        full_doe = []
        for idx_sample in idx_sampling:
            valued_sample = [parameter_grid[attr_row][int(idx)] for attr_row, idx in enumerate(idx_sample)]
            full_doe.append(self.sampled_class(**dict(zip(self.attributes, valued_sample))))
        return full_doe

    def _lhs_sampling(self, samples: int = 10, criterion: str = 'center'):
        varying_sampling = pyDOE.lhs(len(self.sampled_attributes), samples=samples, criterion=criterion)
        full_doe = []
        fixed_values = [attr.value for attr in self.constant_attributes]
        for nodim_sample in varying_sampling:
            varying_values = [attr.dimensionless_to_value(value)
                              for attr, value in zip(self.sampled_attributes, nodim_sample)]
            full_doe.append(self.sampled_class(**dict(zip(self.attributes, fixed_values + varying_values))))
        return full_doe

    def _montecarlo_sampling(self, samples: int = 10):
        full_doe = []
        fixed_values = [attr.value for attr in self.constant_attributes]
        for _ in range(samples):
            varying_values = [random.uniform(attr.min_value, attr.max_value) for attr in self.sampled_attributes]
            full_doe.append(self.sampled_class(**dict(zip(self.attributes, fixed_values + varying_values))))
        return full_doe

    def _get_doe(self, method: str = 'fullfact', samples: int = None, lhs_criterion: str = 'center'):
        if method == 'fullfact':
            return self._full_factorial_sampling()
        if method == 'lhs':
            return self._lhs_sampling(samples=samples, criterion=lhs_criterion)
        if method == 'montecarlo':
            return self._montecarlo_sampling(samples=samples)
        raise NotImplementedError(f"Method '{method}' is not implemented in {self.__class__}._get_doe method.")

    def make_doe(self, method: str = 'fullfact', samples: int = None, lhs_criterion: str = 'center', name: str = ''):
        """
        Generate all `DessiaObject` with the choosen method and store them in a `HeterogeneousList`

        :param method:
            --------
            Method to generate the DOE.
            Can be one of `[fullfact, lhs, montecarlo]`. For more information, see: https://pythonhosted.org/pyDOE/
        :type method: `str`, `optional`, defaults to `'fullfact'`

        :param samples:
            --------
            Targert number of `DessiaObject` in the DOE. Not used for `'lhs'` method.
        :type samples: `int`, `optional`, defaults to `None`,

        :param lhs_criterion:
            --------
            |  Only used with `'lhs'` method.
            |  A string that tells lhs how to sample the points (default: None, which simply randomizes the points \
            |  within the intervals):
            |     - `“center”` or `“c”`: center the points within the sampling intervals
            |     - `“maximin”` or `“m”`: maximize the minimum distance between points, but place the point in a \
                randomized location within its interval
            |     - `“correlation”` or `“corr”`: minimize the maximum correlation coefficient
        :type lhs_criterion: `str`, `optional`, defaults to `'center'`

        :param name: Name of the generated `HeterogeneousList`
        :type name: `str`, `optional`, defaults to `''`

        :return: the `HeterogeneousList` containing all generated samples of the sampled_class
        :rtype: `HeterogeneousList`

        :Examples:
        >>> from dessia_common.sampling import Sampler
        >>> from dessia_common.tests import RandDataD2
        >>> sampled_attr = [BoundedAttributeValue('p_1', 150, 250, 3)]
        >>> constant_attr = [FixedAttributeValue('p_2', 42)]
        >>> randata_sampling = Sampler(RandDataD2, sampled_attributes=sampled_attr, constant_attributes=constant_attr)
        >>> doe_hlist = randata_sampling.make_doe(method='fullfact')
        >>> print(doe_hlist)
        HeterogeneousList 0x7facdb871430: 3 samples, 2 features
        |   Name   |   P_1   |   P_2   |
        --------------------------------
        |          |   150.0 |      42 |
        |          |   200.0 |      42 |
        |          |   250.0 |      42 |

        """
        return HeterogeneousList(self._get_doe(method=method, samples=samples, lhs_criterion=lhs_criterion), name=name)
