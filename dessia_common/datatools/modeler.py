"""
Tools and base classes for machine learning methods.

"""
from typing import List, Dict, Any
from copy import copy
import itertools

from scipy.spatial.distance import pdist, squareform
import numpy as npy
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
    def __init__(self, copy_X: bool = True):
        preprocessing.StandardScaler.__init__(self, with_mean=False, with_std=False, copy=copy_X)

    def fit(self, X: List[List[float]]):
        return

    def fit_transform(self, X: List[List[float]]):
        return X

    def transform(self, X: List[List[float]]):
        return X

    def inverse_transform(self, X: List[List[float]]):
        return X


class Modeler(DessiaObject):

    def __init__(self, scaled_inputs: bool = True, scaled_outputs: bool = False, name: str = ''):
        self.scaled_inputs = scaled_inputs
        self.scaled_outputs = scaled_outputs
        DessiaObject.__init__(self, name=name)

    def _compute_scaler(self, matrix):
        return

    def _scale_matrix(self, matrix):
        return






