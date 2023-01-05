"""
Optimization package for dessia_common.
"""
from typing import List
import cma
import numpy as npy
import scipy.optimize
from dessia_common.core import DessiaObject


class Specifications(DessiaObject):
    """
    Base class to define specifications on parameters of any DessiaObject.
    """
    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)


class FixedAttributeValue(DessiaObject):
    """
    Define a fixed attribute value to run a Design Of Experiment (DOE) or an Optimization.
    """
    _standalone_in_db = True

    def __init__(self, attribute_name: str, value: float, name: str = ''):
        DessiaObject.__init__(self, name=name)
        self.attribute_name = attribute_name
        self.value = value


class BoundedAttributeValue(DessiaObject):
    """
    Define a bounded attribute value to run a Design Of Experiment (DOE) or an Optimization.

    :param attribute_name:
        Name of attribute to bound.
    :type attribute_name: str

    :param min_value:
        Minimum value for this attribute.
    :type min_value: float

    :param max_value:
        Maximum value for this attribute.
    :type max_value: float

    :param number:
        Number of values to generate betwwen those bounds. Only used for sampling.ClassSampler.full_fact method.
    :type number: int

    :param name:
        Name of BoundedAttributeValue.
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, attribute_name: str, min_value: float, max_value: float, number: int = 2, name: str = ''):
        DessiaObject.__init__(self, name=name)
        self.attribute_name = attribute_name
        self.min_value = min_value
        self.max_value = max_value
        self.number = number
        self.interval_length = max_value - min_value

    def dimensionless_to_value(self, dimless_value: float):
        """
        Method to compute the value out of the dimensionless one.
        """
        return self.min_value + dimless_value * self.interval_length

    def dimensionless_value(self, value: float):
        """
        Method to compute the dimensionless value out of the dimensioned one.
        """
        return (value - self.min_value) / self.interval_length


class Optimizer(DessiaObject):
    """
    Base class for creating an Optimizer.
    """

    def adimensioned_vector(self, x):
        """
        Returns the adimensioned vector from the real one.
        """

    def reduced_vector(self, x):
        """
        Get reduced vector of vector x.
        """

    def cma_bounds(self):
        """
        Returns the bounds in the CMA format.
        """

    def scipy_minimize_bounds(self):
        """
        Minimize value between bounds.
        """

    def cma_optimization(self):
        """
        Runs an optimization of the model with CMA method.
        """

    def scipy_minimize_optimization(self):
        """
        Runs an optimization of the model with scipy gradient method.
        """


class DrivenModelOptimizer(Optimizer):
    """
    Abstract class for Optimizer driven with a model.
    """

    def __init__(self, model, name: str = ''):
        Optimizer.__init__(self, name=name)
        self.model = model

    def get_model_from_vector(self):
        """
        Get model from vector.
        """ #TODO: change docstring
        # modify inplace model from vector
        raise NotImplementedError('the method must be overloaded by subclassing class')


class InstantiatingModelOptimizer(Optimizer):
    """
    Abstract class, to be subclassed by real class. Instantiate a new model at each point request.
    """

    def __init__(self, fixed_parameters: List[FixedAttributeValue], optimization_bounds: List[BoundedAttributeValue],
                 name: str = ''):
        self.fixed_parameters = fixed_parameters
        self.optimization_bounds = optimization_bounds
        Optimizer.__init__(self, name=name)

        self.number_parameters = len(self.optimization_bounds)

    def instantiate_model(self, attributes_values):
        """
        Instantiate model to compute cost function of Optimizer.
        """
        raise NotImplementedError('the method instantiate_model must be overloaded by subclassing class')

    def dimensionless_vector_to_vector(self, dl_vector):
        """
        Returns the vector from the adimensioned one.
        """
        return [bound.dimensionless_to_value(dl_xi) for dl_xi, bound in zip(dl_vector, self.optimization_bounds)]

    def vector_to_attributes_values(self, vector: List[float]):
        """
        Returns a dict of attribute_name: value for each parameter name.
        """
        attributes = {fp.attribute_name: fp.value for fp in self.fixed_parameters}

        for bound, xi in zip(self.optimization_bounds, vector):
            attributes[bound.attribute_name] = xi
        return attributes

    def objective_from_dimensionless_vector(self, dl_vector):
        """
        Compute the real values of objective attributes of object from their optimized dimensionless values.
        """
        attributes_values = self.vector_to_attributes_values(self.dimensionless_vector_to_vector(dl_vector))
        model = self.instantiate_model(attributes_values)
        return self.objective_from_model(model)

    def objective_from_model(self, model, clearance: float = 0.003):
        """
        Compute cost of current configuration with model methods.
        """
        raise NotImplementedError('the method objective_from_model must be overloaded by subclassing class')

    def scipy_bounds(self):
        """
        Returns the bounds in the scipy format.
        """
        return [(0, 1) for b in self.optimization_bounds]

    def cma_bounds(self):
        """
        Returns the bounds in the CMA format.
        """
        return [[0] * len(self.optimization_bounds),
                [1] * len(self.optimization_bounds)]

    def optimize_gradient(self, method: str = 'L-BFGS-B', x0: List[float] = None):
        """
        Optimize the problem by gradient methods from scipy.
        """
        if x0 is None:
            x0 = npy.random.random(self.number_parameters)
        bounds = self.scipy_bounds()

        result = scipy.optimize.minimize(self.objective_from_dimensionless_vector, x0, bounds=bounds, method=method)

        attributes_values = self.vector_to_attributes_values(self.dimensionless_vector_to_vector(result.x))
        model = self.instantiate_model(attributes_values)
        return model, result.fun

    def optimize_cma(self):
        """
        Optimize the problem using the CMA-ES algorithm.
        """
        x0 = npy.random.random(self.number_parameters)
        bounds = self.cma_bounds()
        xra, fx_opt = cma.fmin(self.objective_from_dimensionless_vector, x0, 0.6, options={'bounds': bounds,
                                                                                           'tolfun': 1e-3,
                                                                                           'maxiter': 250,
                                                                                           'verbose': -9,
                                                                                           'ftarget': 0.2})[0:2]

        attributes_values = self.vector_to_attributes_values(self.dimensionless_vector_to_vector(xra))

        model = self.instantiate_model(attributes_values)
        return model, fx_opt

    def optimize_cma_then_gradient(self, method: str = 'L-BFGS-B'):
        """
        Optimize the problem by combining a first phase of CMA for global optimum search and gradient for polishing.
        """
        model_cma, fx_cma = self.optimize_cma()
        x0 = npy.array([getattr(model_cma, attr.attribute_name) for attr in self.optimization_bounds])
        model, best_fx = self.optimize_gradient(method=method, x0=x0)
        if fx_cma <= best_fx:
            model = model_cma
            best_fx = fx_cma
        return model, best_fx
