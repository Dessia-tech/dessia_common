

import dessia_common.core as dc


class Specifications(dc.DessiaObject):
    def __init__(self, name:str=''):
        self.name = name

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

    def __init__(self, model):
        self.model = model

    def get_model_from_vector(self):
        # modify inplace model from vector
        raise NotImplementedError('the method must be overloaded by subclassing class')

class InstantiatingModelOptimizer(Optimizer):
    """
    Abstract class, to be subclassed by real class
    Instantiate a new model at each at each point request
    """

    def __init__(self):
        # self.model = model
        pass


    def get_model_from_vector(self):
        # instantiate a model from vector
        raise NotImplementedError('the method must be overloaded by subclassing class')