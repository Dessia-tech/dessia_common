

import dessia_common.core as dc


class Specifications(dc.DessiaObject):
    def __init__(self, name:str=''):
        self.name = name

class FixedAttributeValue(dc.DessiaObject):
    _standalone_in_db = True

    def __init__(self, attribute_name:str, value:float,
                 name:str=''):
        self.name = name
        self.attribute_name = attribute_name
        self.value = value


class BoundedAttributeValue(dc.DessiaObject):
    _standalone_in_db = True

    def __init__(self, attribute_name:str, min_value:float, max_value:float,
                 name:str=''):
        self.name = name
        self.attribute_name = attribute_name
        self.min_value = min_value
        self.max_value = max_value
        self.interval_length = max_value - min_value

    def dimensionless_to_value(self, dimless_value:float):
        return self.min_value + dimless_value*self.interval_length

    def dimensionless_value(self, value:float):
        return (value-self.min_value)/self.interval_length

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