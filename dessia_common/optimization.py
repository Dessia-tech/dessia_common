"""
Optimization package for dessia_common

"""
from typing import List
import cma
import numpy as npy
import scipy.optimize
import dessia_common.core as dc


class Specifications(dc.DessiaObject):
    def __init__(self, name: str = ''):
        dc.DessiaObject.__init__(self, name=name)


class FixedAttributeValue(dc.DessiaObject):
    _standalone_in_db = True

    def __init__(self, attribute_name: str, value: float,
                 name: str = ''):
        dc.DessiaObject.__init__(self, name=name)
        self.attribute_name = attribute_name
        self.value = value


class BoundedAttributeValue(dc.DessiaObject):
    _standalone_in_db = True

    def __init__(self, attribute_name: str, min_value: float, max_value: float,
                 name: str = ''):
        dc.DessiaObject.__init__(self, name=name)
        self.attribute_name = attribute_name
        self.min_value = min_value
        self.max_value = max_value
        self.interval_length = max_value - min_value

    def dimensionless_to_value(self, dimless_value: float):
        return self.min_value + dimless_value * self.interval_length

    def dimensionless_value(self, value: float):
        return (value - self.min_value) / self.interval_length


class Optimizer(dc.DessiaObject):
    """
    Common parts of optimizers
    """

    def adimensioned_vector(self, x):
        pass

    def reduced_vector(self, x):
        pass

    def cma_bounds(self):
        pass

    def scipy_minimize_bounds(self):
        pass

    def cma_optimization(self):
        pass

    def scipy_minimize_optimization(self):
        pass


class DrivenModelOptimizer(Optimizer):
    """
    Abstract class
    """

    def __init__(self, model, name: str = ''):
        Optimizer.__init__(self, name=name)
        self.model = model

    def get_model_from_vector(self):
        # modify inplace model from vector
        raise NotImplementedError('the method must be overloaded by subclassing class')


class InstantiatingModelOptimizer(Optimizer):
    """
    Abstract class, to be subclassed by real class
    Instantiate a new model at each at each point request
    """

    def __init__(self, fixed_parameters: List[FixedAttributeValue],
                 optimization_bounds: List[BoundedAttributeValue],
                 name: str = ''):
        self.fixed_parameters = fixed_parameters
        self.optimization_bounds = optimization_bounds
        Optimizer.__init__(self, name=name)

        self.number_parameters = len(self.optimization_bounds)

    def instantiate_model(self, attributes_values):
        raise NotImplementedError('the method instantiate_model must be overloaded by subclassing class')

    def dimensionless_vector_to_vector(self, dl_vector):
        return [bound.dimensionless_to_value(dl_xi) for dl_xi, bound in zip(dl_vector, self.optimization_bounds)]

    def vector_to_attributes_values(self, vector: List[float]):
        attributes = {fp.attribute_name: fp.value for fp in self.fixed_parameters}

        for bound, xi in zip(self.optimization_bounds, vector):
            attributes[bound.attribute_name] = xi
        return attributes

    def objective_from_dimensionless_vector(self, dl_vector):
        attributes_values = self.vector_to_attributes_values(self.dimensionless_vector_to_vector(dl_vector))
        model = self.instantiate_model(attributes_values)
        return self.objective_from_model(model)

    def objective_from_model(self, model, clearance: float = 0.003):
        raise NotImplementedError('the method objective_from_model must be overloaded by subclassing class')

    def scipy_bounds(self):
        # return [(b.min_value, b.max_value) for b in self.optimization_bounds]
        return [(0, 1) for b in self.optimization_bounds]

    def cma_bounds(self):
        return [[0] * len(self.optimization_bounds),
                [1] * len(self.optimization_bounds)]

    def optimize_gradient(self, method: str = 'L-BFGS-B'):
        x0 = npy.random.random(self.number_parameters)
        bounds = self.scipy_bounds()
        result = scipy.optimize.minimize(self.objective_from_dimensionless_vector,
                                         x0, bounds=bounds, method=method)

        attributes_values = self.vector_to_attributes_values(
            self.dimensionless_vector_to_vector(result.x))

        model = self.instantiate_model(attributes_values)
        return model, result.fun

    def optimize_cma(self):

        x0 = npy.random.random(self.number_parameters)

        bounds = self.cma_bounds()
        xra, fx_opt = cma.fmin(self.objective_from_dimensionless_vector,
                               x0, 0.6, options={'bounds': bounds,
                                                 'tolfun': 1e-3,
                                                 'maxiter': 250,
                                                 'verbose': 0,
                                                 'ftarget': 0.2})[0:2]

        attributes_values = self.vector_to_attributes_values(
            self.dimensionless_vector_to_vector(xra))

        model = self.instantiate_model(attributes_values)

        return model, fx_opt
