""" Library for sampling data. """
from typing import List, Type

import random
import numpy as npy
import pyDOE2 as pyDOE

from dessia_common.core import DessiaObject
from dessia_common.datatools.dataset import Dataset
from dessia_common.optimization import FixedAttributeValue, BoundedAttributeValue


class ClassSampler(DessiaObject):
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

    def full_factorial(self):
        """
        Generate all `DessiaObject` with a Full Factorial sampling and store them in a `Dataset`.

        A number to discretize each dimension of the problem must be specified in used `BoundedAttributeValue` and they
        are the only ones to be used for sampling data.

        :return: a `Dataset` containing all generated samples of the sampled_class
        :rtype: `Dataset`
        """
        instances_numbers = self._get_instances_numbers()
        parameter_grid = self._build_parameter_grid(instances_numbers)
        idx_sampling = pyDOE.fullfact(instances_numbers)
        full_doe = []
        for idx_sample in idx_sampling:
            valued_sample = [parameter_grid[attr_row][int(idx)] for attr_row, idx in enumerate(idx_sample)]
            full_doe.append(self.sampled_class(**dict(zip(self.attributes, valued_sample))))
        return full_doe

    def lhs(self, samples: int, criterion: str):
        """
        Generate all `DessiaObject` with a Latin Hypercube Sampling (LHS) and store them in a `Dataset`.

        Documentation LHS: https://www.statology.org/latin-hypercube-sampling/

        :param samples:
            Targert number of `DessiaObject` in the DOE.
        :type samples: `int`

        :return: a `Dataset` containing all generated samples of the sampled_class
        :rtype: `Dataset`
        """
        varying_sampling = pyDOE.lhs(len(self.sampled_attributes), samples=samples, criterion=criterion)
        full_doe = []
        fixed_values = [attr.value for attr in self.constant_attributes]
        for nodim_sample in varying_sampling:
            varying_values = [attr.dimensionless_to_value(value)
                              for attr, value in zip(self.sampled_attributes, nodim_sample)]
            full_doe.append(self.sampled_class(**dict(zip(self.attributes, fixed_values + varying_values))))
        return full_doe

    def montecarlo(self, samples: int):
        """
        Generate all `DessiaObject` with a Monte-Carlo (random uniform) sampling and store them in a `Dataset`.

        In other words, it is a random uniform sampling for all dimensions of the problem.
        For example, a two dimensions problem sampled with 100 elements with Monte-Carlo method would result in 100
        samples. Each dimension of each sample would have a random uniform value. Redundancies are allowed with this
        sampling method.

        :param samples:
            Targert number of `DessiaObject` in the DOE.
        :type samples: `int`

        :return: a `Dataset` containing all generated samples of the sampled_class
        :rtype: `Dataset`
        """
        full_doe = []
        fixed_values = [attr.value for attr in self.constant_attributes]
        for _ in range(samples):
            varying_values = [random.uniform(attr.min_value, attr.max_value) for attr in self.sampled_attributes]
            full_doe.append(self.sampled_class(**dict(zip(self.attributes, fixed_values + varying_values))))
        return full_doe

    def _get_doe(self, method: str, samples: int, lhs_criterion: str):
        if method == 'fullfact':
            return self.full_factorial()
        if method == 'lhs':
            return self.lhs(samples=samples, criterion=lhs_criterion)
        if method == 'montecarlo':
            return self.montecarlo(samples=samples)
        raise NotImplementedError(f"Method '{method}' is not implemented in {self.__class__}._get_doe method.")

    def make_doe(self, samples: int, method: str = 'fullfact', lhs_criterion: str = 'center', name: str = ''):
        """
        Generate all `DessiaObject` with the choosen method and store them in a `Dataset`.

        :param samples: Targets number of `DessiaObject` in the DOE. Not used for `'fullfact'` method.
        :type samples: `int`

        :param method:
            Method to generate the DOE.
            Can be one of `[fullfact, lhs, montecarlo]`. For more information, see: https://pythonhosted.org/pyDOE/
        :type method: `str`, `optional`, defaults to `'fullfact'`

        :param lhs_criterion:
            |  Only used with `'lhs'` method.
            |  A string that tells lhs how to sample the points (default: None, which simply randomizes the points \
            |  within the intervals):
            |     - `“center”` or `“c”`: center the points within the sampling intervals
            |     - `“maximin”` or `“m”`: maximize the minimum distance between points, but place the point in a \
                randomized location within its interval
            |     - `“correlation”` or `“corr”`: minimize the maximum correlation coefficient
        :type lhs_criterion: `str`, `optional`, defaults to `'center'`

        :param name: Name of the generated `Dataset`
        :type name: `str`, `optional`, defaults to `''`

        :return: the `Dataset` containing all generated samples of the sampled_class
        :rtype: `Dataset`

        :Examples:
        >>> from dessia_common.datatools.sampling import ClassSampler
        >>> from dessia_common.tests import RandDataD2
        >>> sampled_attr = [BoundedAttributeValue('p_1', 150, 250, 3)]
        >>> constant_attr = [FixedAttributeValue('p_2', 42)]
        >>> randata_sampling = Sampler(RandDataD2, sampled_attributes=sampled_attr, constant_attributes=constant_attr)
        >>> doe_hlist = randata_sampling.make_doe(method='fullfact')
        >>> print(doe_hlist)
        Dataset 0x7facdb871430: 3 samples, 2 features
        |   Name   |   P_1   |   P_2   |
        --------------------------------
        |          |   150.0 |      42 |
        |          |   200.0 |      42 |
        |          |   250.0 |      42 |
        """
        return Dataset(self._get_doe(method=method, samples=samples, lhs_criterion=lhs_criterion), name=name)
