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
from dessia_common.core import DessiaObject


class IdentityScaler(DessiaObject):
    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    def _instantiate(self):
        return preprocessing.StandardScaler(with_mean = False, with_std = False)

    @classmethod
    def fit(cls, matrix: List[List[float]], name: str = ''):
        return cls(name)

    def transform(self, matrix: List[List[float]]):
        return matrix

    @classmethod
    def fit_transform(cls, matrix: List[List[float]], name: str = ''):
        return cls(name), matrix


class StandardScaler(IdentityScaler):
    def __init__(self, mean_: List[float] = None, scale_: List[float] = None, var_: List[float] = None, name: str = ''):
        self.mean_ = mean_
        self.scale_ = scale_
        self.var_ = var_
        IdentityScaler.__init__(self, name=name)

    def _instantiate(self):
        scaler = preprocessing.StandardScaler()
        scaler.mean_ = self.mean_
        scaler.scale_ = self.scale_
        scaler.std_ = self.std_
        return scaler

    @classmethod
    def fit(cls, matrix: List[List[float]], name: str = ''):
        scaler = preprocessing.StandardScaler()
        scaler.fit(matrix)
        return cls(scaler.mean_, scaler.scale_, scaler.var_, name)

    def transform(self, matrix: List[List[float]]):
        scaler = self._instantiate()
        return scaler.transform(matrix)

    @classmethod
    def fit_transform(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls.fit(matrix, name)
        return cls(scaler.mean_, scaler.scale_, scaler.var_, name), scaler.transform(matrix)


class DessiaModel(DessiaObject):
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
        self._required_attributes = []
        self.scaled_inputs = scaled_inputs
        self.scaled_outputs = scaled_outputs
        DessiaObject.__init__(self, name=name)

##### SCALE ##########
    def _initialize_scaler(self, is_scaled: bool):
        if is_scaled:
            return preprocessing.StandardScaler()
        return IdentityScaler()

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






