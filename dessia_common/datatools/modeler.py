"""
Tools and base classes for machine learning methods.

"""
from typing import List, Dict, Any
from copy import copy
import itertools

from scipy.spatial.distance import pdist, squareform
import numpy as npy
import sklearn
from sklearn import preprocessing

try:
    from plot_data.core import Scatter, Histogram, MultiplePlots, Tooltip, ParallelPlot, PointFamily, EdgeStyle, Axis, \
        PointStyle
    from plot_data.colors import BLUE, GREY
except ImportError:
    pass
from dessia_common.core import DessiaObject, DessiaFilter, FiltersList, templates
from dessia_common.datatools.metrics import mean, std, variance, covariance_matrix


class IdentityScaler(preprocessing.StandardScaler):
    def __init__(self, copy_matrix: bool = True):
        preprocessing.StandardScaler.__init__(self, with_mean=False, with_std=False, copy=copy_matrix)

    def fit(self, matrix: List[List[float]]):
        return

    def fit_transform(self, matrix: List[List[float]]):
        return matrix

    def transform(self, matrix: List[List[float]]):
        return matrix

    def inverse_transform(self, matrix: List[List[float]]):
        return matrix


class BaseModel(DessiaObject):
    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    def fit(self, matrix: List[List[float]]):
        return

    def fit_predict(self, matrix: List[List[float]]):
        return matrix

    def predict(self, matrix: List[List[float]]):
        return matrix


class Modeler(DessiaObject):
    def __init__(self, scaled_inputs: bool = True, scaled_outputs: bool = False, name: str = ''):
        self.model_ = dict()
        self.model_attributes = dict()
        self.scaled_inputs = scaled_inputs
        self.scaled_outputs = scaled_outputs
        DessiaObject.__init__(self, name=name)

##### SCALE ##########
    def _init_scaler(self, is_scaled: bool):
        if is_scaled:
            return preprocessing.StandardScaler()
        return IdentityScaler()

    def _fit_scaler(self, matrix: List[List[float]], scaler):
        scaler.fit(matrix)

    def _scale_matrix(self, matrix: List[List[float]], scaler):
        return scaler.transform(matrix)

    def _fit_transform_scaler(self, matrix: List[List[float]], scaler):
        return scaler.fit_transform(matrix)

    def _auto_scale(self, matrix: List[List[float]], is_scaled: bool):
        scaler = self._init_scaler(is_scaled)
        scaled_matrix = self._fit_transform_scaler(matrix, scaler)
        return scaler, scaled_matrix

##### MODEL ##########
    def _init_model(self):
        return BaseModel()

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
        return self.model.fit(scaled_inputs, scaled_outputs)

    def predict(self, inputs: List[List[float]], input_scaler, output_scaler):
        scaled_inputs = input_scaler.transform(inputs)
        return output_scaler.inverse_transform(self.model.predict(scaled_inputs))

    def fit_predict(self, inputs: List[List[float]], outputs: List[List[float]]):
        input_scaler, scaled_inputs = self._auto_scale(inputs, self.scaled_inputs)
        output_scaler, scaled_outputs = self._auto_scale(outputs, self.scaled_outputs)
        return self.model.fit_predit(scaled_inputs, scaled_outputs)

# Does sklearn objects have to be serialized ?






