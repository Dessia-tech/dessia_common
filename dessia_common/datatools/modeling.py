"""
Tools and base classes for machine learning methods.

"""
from typing import List, Dict, Any
from copy import copy
import itertools

from scipy.spatial.distance import pdist, squareform
import numpy as npy
import sklearn
from sklearn import preprocessing, linear_model

try:
    from plot_data.core import Scatter, Histogram, MultiplePlots, Tooltip, ParallelPlot, PointFamily, EdgeStyle, Axis, \
        PointStyle
    from plot_data.colors import BLUE, GREY
except ImportError:
    pass
from dessia_common.core import DessiaObject

# =============================================================================
# Scalers
# =============================================================================
class DessiaScaler(DessiaObject):
    _rebuild_attributes = []

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _call_skl_scaler(cls):
        raise NotImplementedError('Method _call_skl_scaler not implemented for DessiaScaler. Please use children.')

    def _instantiate_skl_scaler(self):
        scaler = self._call_skl_scaler()
        for attr in self._rebuild_attributes:
            setattr(scaler, attr, getattr(self, attr))
        return scaler

    @classmethod
    def _instantiate_dessia_scaler(cls, scaler, name: str = ''):
        kwargs_dict = {'name': name}
        for attr in cls._rebuild_attributes:
            if hasattr(scaler, attr):
                if isinstance(getattr(scaler, attr), npy.ndarray):
                    kwargs_dict[attr] = getattr(scaler, attr).tolist()
                kwargs_dict[attr] = getattr(scaler, attr)
        return kwargs_dict

    @classmethod
    def fit(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls._call_skl_scaler()
        scaler.fit(matrix)
        return cls(**cls._instantiate_dessia_scaler(scaler, name))

    def transform(self, matrix: List[List[float]]):
        scaler = self._instantiate_skl_scaler()
        return scaler.transform(matrix).tolist()

    @classmethod
    def fit_transform(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls.fit(matrix, name)
        return scaler, scaler.transform(matrix)


class StandardScaler(DessiaScaler):
    _rebuild_attributes = ['mean_', 'scale_', 'var_']
    _standalone_in_db = True

    def __init__(self, mean_: List[float] = None, scale_: List[float] = None, var_: List[float] = None, name: str = ''):
        self.mean_ = mean_
        self.scale_ = scale_
        self.var_ = var_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _call_skl_scaler(cls):
        return preprocessing.StandardScaler()


class IdentityScaler(StandardScaler):

    def __init__(self, mean_: List[float] = None, scale_: List[float] = None, var_: List[float] = None, name: str = ''):
        StandardScaler.__init__(self, mean_=mean_, scale_=scale_, var_=var_, name=name)

    @classmethod
    def _call_skl_scaler(cls):
        return preprocessing.StandardScaler(with_mean = False, with_std = False)

# =============================================================================
# Models
# =============================================================================
class DessiaModel(DessiaObject):
    _rebuild_attributes = []

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _call_skl_model(cls):
        raise NotImplementedError('Method _call_skl_model not implemented for DessiaModel. Please use children.')

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        for attr in self._rebuild_attributes:
            setattr(model, attr, getattr(self, attr))
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        for attr in cls._rebuild_attributes:
            if hasattr(model, attr):
                if isinstance(getattr(model, attr), npy.ndarray):
                    kwargs_dict[attr] = getattr(model, attr).tolist()
                kwargs_dict[attr] = getattr(model, attr)
        return kwargs_dict

    @classmethod
    def fit(cls, matrix: List[List[float]], name: str = ''):
        model = cls._call_skl_model()
        model.fit(matrix)
        return cls(**cls._instantiate_dessia_model(model, name))

    def predict(self, matrix: List[List[float]]):
        model = self._instantiate_skl_model()
        return model.predict(matrix).tolist()

    @classmethod
    def fit_predict(cls, matrix: List[List[float]], name: str = ''):
        model = cls.fit(matrix, name)
        return model, model.predict(matrix)


class LinearRegression(DessiaModel):
    _rebuild_attributes = ['coefs_', 'intercept_']

    def __init__(self, coefs_: List[List[float]] = None, intercept_: List[List[float]] = None, name: str = ''):
        self.coefs_ = coefs_
        self.intercept_ = intercept_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _call_skl_model(cls):
        return linear_model.Ridge()

    # @staticmethod
    # def _compile_model_attributes(alpha: float = 1., fit_intercept: bool = True, tol: float = 0.001,
    #                               solver: str = 'auto'):
    #     return {'alpha': alpha, 'fit_intercept': fit_intercept, 'tol': tol, 'solver': solver}

    # @classmethod
    # def fit(cls, matrix: List[List[float]], alpha: float = 1., fit_intercept: bool = True, tol: float = 0.001,
    #         solver: str = 'auto', name: str = ''):
    #     model_attributes = cls._compile_model_attributes(alpha, fit_intercept, tol, solver)
    #     regressor = linear_model.Ridge(**model_attributes)
    #     regressor.fit(matrix)
    #     return cls(coefs_=regressor.coefs_, intercept_=regressor.intercept__, name=name)

    # def predict(self, matrix: List[List[float]]):
    #     model = self._instantiate(linear_model.Ridge())
    #     return model.predict(matrix).tolist()

    # @classmethod
    # def fit_predict(cls, matrix: List[List[float]], alpha: float = 1., fit_intercept: bool = True, tol: float = 0.001,
    #                 solver: str = 'auto', name: str = ''):
    #     model = cls.fit(matrix, alpha=alpha, fit_intercept=fit_intercept, tol=tol, solver=solver, name=name)
    #     return model, model.predict(matrix)


# =============================================================================
# Modeler
# =============================================================================
class Modeler(DessiaObject):
    def __init__(self, scaled_inputs: bool = True, scaled_outputs: bool = False, name: str = ''):
        self.model_ = dict()
        self.model_attributes = dict()
        self._required_attributes = []
        self.scaled_inputs = scaled_inputs
        self.scaled_outputs = scaled_outputs
        DessiaObject.__init__(self, name=name)

##### SCALE ##########
    def _initialize_scaler(self, is_scaled: bool):
        if is_scaled:
            return StandardScaler()
        return IdentityScaler()

##### MODEL ##########
    def _initialize_model(self):
        return DessiaModel()

    def _set_model_attributes(self, model, attributes: Dict[str, float]):
        for attr, value in attributes.items():
            setattr(model, attr, value)
        return model

    def _instantiate_model(self):
        model = self._init_model()
        model = self._set_model_attributes(model, self.model_attributes)
        model = self._set_model_attributes(model, self.model_)
        return model

##### MODEL METHODS ##########
    def fit(self, inputs: List[List[float]], outputs: List[List[float]]):
        input_scaler, scaled_inputs = self._auto_scale(inputs, self.scaled_inputs)
        output_scaler, scaled_outputs = self._auto_scale(outputs, self.scaled_outputs)
        model = self._instantiate_model()
        model.fit(scaled_inputs, scaled_outputs)
        self.model_ = {key: value for key, value in model.items() if key in self._required_attributes}

    def predict(self, inputs: List[List[float]], input_scaler, output_scaler):
        scaled_inputs = input_scaler.transform(inputs)
        model = self._instantiate_model()
        return output_scaler.inverse_transform(model.predict(scaled_inputs))

    def fit_predict(self, inputs: List[List[float]], outputs: List[List[float]]):
        input_scaler, scaled_inputs = self._auto_scale(inputs, self.scaled_inputs)
        output_scaler, scaled_outputs = self._auto_scale(outputs, self.scaled_outputs)
        model = self._instantiate_model()
        predicted_outputs = model.fit_predict(scaled_inputs, scaled_outputs)
        self.model_ = {key: value for key, value in model.items() if key in self._required_attributes}
        return predicted_outputs

# Does sklearn objects have to be serialized ?






