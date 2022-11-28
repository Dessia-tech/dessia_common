"""
Tools and base classes for machine learning methods.

"""
from typing import List, Dict, Any, Tuple
from copy import copy
import itertools

from scipy.spatial.distance import pdist, squareform
import numpy as npy
import sklearn
from sklearn import preprocessing, linear_model, ensemble, tree

try:
    from plot_data.core import Scatter, Histogram, MultiplePlots, Tooltip, ParallelPlot, PointFamily, EdgeStyle, Axis, \
        PointStyle
    from plot_data.colors import BLUE, GREY
except ImportError:
    pass
from dessia_common.core import DessiaObject


# ======================================================================================================================
#                                                     S C A L E R S
# ======================================================================================================================
class DessiaScaler(DessiaObject): # TODO: is there a better name ?
    _rebuild_attributes = []

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for DessiaScaler. Please use children.')

    def _call_skl_scaler(self):
        return self._skl_class()()

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
        return cls(**kwargs_dict)

    @classmethod
    def fit(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls._skl_class()()
        scaler.fit(matrix)
        return cls._instantiate_dessia_scaler(scaler, name)

    def transform(self, matrix: List[List[float]]):
        scaler = self._instantiate_skl_scaler()
        return scaler.transform(matrix).tolist()

    @classmethod
    def fit_transform(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls.fit(matrix, name)
        return scaler, scaler.transform(matrix)


class StandardScaler(DessiaScaler):
    """
    Data scaler that standardly scale data. The operation made by this scaler is `new_X = (X - mean(X))/std(X)`.

    :param mean_:
        --------
        List of means
    :type mean_: `List[float]`, `optional`, defaults to `None`

    :param scale_:
        --------
        List of standard deviations
    :type mean_: `List[float]`, `optional`, defaults to `None`

    :param var_:
        --------
        List of variances
    :type mean_: `List[float]`, `optional`, defaults to `None`

    """
    _rebuild_attributes = ['mean_', 'scale_', 'var_']
    _standalone_in_db = True

    def __init__(self, mean_: List[float] = None, scale_: List[float] = None, var_: List[float] = None, name: str = ''):
        self.mean_ = mean_
        self.scale_ = scale_
        self.var_ = var_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return preprocessing.StandardScaler


class IdentityScaler(StandardScaler):
    """
    Data scaler that does no scaler. It is implemented to keep code consistency and readability and may be useful to
    avoid conditions if data is scaled or not.

    :param mean_:
        --------
        List of means
    :type mean_: `None`

    :param scale_:
        --------
        List of standard deviations
    :type mean_: `None`

    :param var_:
        --------
        List of variances
    :type mean_: `None`

    """
    def __init__(self, mean_: List[float] = None, scale_: List[float] = None, var_: List[float] = None, name: str = ''):
        StandardScaler.__init__(self, mean_=mean_, scale_=scale_, var_=var_, name=name)

    def _call_skl_scaler(self):
        return self._skl_class()(with_mean = False, with_std = False)



# ======================================================================================================================
#                                                        M O D E L S
# ======================================================================================================================
class DessiaModel(DessiaObject): # TODO: is there a better name ?

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for DessiaModel. Please use children.')

    def _call_skl_model(self):
        return self._skl_class()()

    def _instantiate_skl_model(self):
        raise NotImplementedError(f'Method _instantiate_skl_model not implemented for {type(self).__name__}.')

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        raise NotImplementedError(f'Method _instantiate_dessia_model not implemented for {cls.__name__}.')

    @classmethod
    def fit_(cls, inputs: List[List[float]], outputs: List[List[float]], name: str = '', **hyperparameters):
        model = cls._skl_class()(**hyperparameters)
        model.fit(inputs, outputs)
        return cls._instantiate_dessia_model(model, name)

    def predict(self, inputs: List[List[float]]):
        model = self._instantiate_skl_model()
        return model.predict(inputs).tolist()

    @classmethod
    def fit_predict_(cls, inputs: List[List[float]], outputs: List[List[float]], predicted_inputs: List[List[float]],
                    name: str = '', **hyperparameters):
        model = cls.fit_(inputs, outputs, name, **hyperparameters)
        return model, model.predict(predicted_inputs)


class LinearRegression(DessiaModel):
    _standalone_in_db = True

    def __init__(self, coef_: List[List[float]] = None, intercept_: List[List[float]] = None, name: str = ''):
        self.coef_ = coef_
        self.intercept_ = intercept_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return linear_model.Ridge

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        model.coef_ = npy.array(self.coef_)
        model.intercept_  = npy.array(self.intercept_)
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        return cls(coef_=model.coef_.tolist(), intercept_=model.intercept_.tolist(), name=name)

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[List[float]], name: str = '',
            alpha: float = 1., fit_intercept: bool = True, tol: float = 0.001):
        return cls.fit_(inputs, outputs, name=name, alpha=alpha, fit_intercept=fit_intercept, tol=tol)

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[List[float]], predicted_inputs: List[List[float]],
                    name: str = '', alpha: float = 1., fit_intercept: bool = True, tol: float = 0.001):
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name,
                                alpha=alpha, fit_intercept=fit_intercept, tol=tol)


class DessiaTree(DessiaModel): # TODO: is there a better name ?
    _standalone_in_db = True

    def __init__(self, n_classes: List[int], n_features: int, n_outputs: int, tree_state: Dict[str, Any],
                 name: str = ''):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.tree_state = tree_state
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return tree._tree.Tree

    def _call_skl_model(self):
        return self._skl_class()(self.n_features, npy.array(self.n_classes), self.n_outputs)

    @staticmethod
    def _getstate_dessia(model):
        state = model.__getstate__()
        dessia_state = {'max_depth': int(state['max_depth'])}
        dessia_state['node_count'] = int(state['node_count'])
        dessia_state['values'] = state['values'].tolist()
        dessia_state['nodes'] = {'dtypes': state['nodes'].dtype.descr, 'values': state['nodes'].tolist()}
        return dessia_state

    @staticmethod
    def _setstate_dessia(model, state):
        skl_state = dict()
        skl_state = {'max_depth': int(state['max_depth'])}
        skl_state['node_count'] = int(state['node_count'])
        skl_state['values'] = npy.array(state['values'])
        skl_state['nodes'] = npy.array(state['nodes']['values'], dtype=state['nodes']['dtypes'])
        model.__setstate__(skl_state)
        return model

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        model = self._setstate_dessia(model, self.tree_state)
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['tree_state'] = cls._getstate_dessia(model)
        kwargs_dict['n_classes'] = model.n_classes.tolist()
        kwargs_dict['n_features'] = model.n_features
        kwargs_dict['n_outputs'] = model.n_outputs
        return cls(**kwargs_dict)

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[List[float]], name: str = ''):
        raise NotImplementedError('fit method is not supposed to be used in DessiaTree and is not implemented.')

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[List[float]], predicted_inputs: List[List[float]],
                    name: str = ''):
        raise NotImplementedError('fit_predict method is not supposed to be used in DessiaTree and is not implemented.')


class DecisionTree(DessiaModel):
    _standalone_in_db = True

    def __init__(self, tree_: DessiaTree = None, name: str = ''):
        self.tree_ = tree_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[List[float]], name: str = '',
            criterion: str = 'squared_error', max_depth: int = None):
        if cls is not DecisionTreeRegressor and criterion == 'squared_error':
            criterion = 'gini'
        return cls.fit_(inputs, outputs, name=name, criterion=criterion, max_depth=max_depth)

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[List[float]], predicted_inputs: List[List[float]],
                    name: str = '', criterion: str = 'squared_error', max_depth: int = None):
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, criterion=criterion, max_depth=max_depth)


class DecisionTreeRegressor(DecisionTree):
    _standalone_in_db = True

    def __init__(self, n_outputs_: int, tree_: DessiaTree = None, name: str = ''):
        self.n_outputs_ = n_outputs_
        DecisionTree.__init__(self, tree_=tree_, name=name)

    @classmethod
    def _skl_class(cls):
        return tree.DecisionTreeRegressor

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        model.n_outputs_ = self.n_outputs_
        model.tree_ = self.tree_._instantiate_skl_model()
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['tree_'] = DessiaTree._instantiate_dessia_model(model.tree_)
        kwargs_dict['n_outputs_'] = model.n_outputs_
        return cls(**kwargs_dict)


class DecisionTreeClassifier(DecisionTreeRegressor):
    def __init__(self, n_classes_: int, n_outputs_: int, tree_: DessiaTree = None, name: str = ''):
        self.n_classes_ = n_classes_
        DecisionTreeRegressor.__init__(self, n_outputs_=n_outputs_, tree_=tree_, name=name)

    @classmethod
    def _skl_class(cls):
        return tree.DecisionTreeClassifier

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        model.n_outputs_ = self.n_outputs_
        model.n_classes_ = self.n_classes_
        model.tree_ = self.tree_._instantiate_skl_model()
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['tree_'] = DessiaTree._instantiate_dessia_model(model.tree_)
        kwargs_dict['n_outputs_'] = model.n_outputs_
        kwargs_dict['n_classes_'] = model.n_classes_
        return cls(**kwargs_dict)


class RandomForest(DessiaModel):
    def __init__(self, estimators_: List[List[float]] = None,
                 name: str = ''):
        self.estimators_ = estimators_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for RandomForest. Please use '\
                                  'RandomForestClassifier or RandomForestRegressor.')

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[List[float]], name: str = '',
            n_estimators: int = 100, criterion: str = 'squared_error', max_depth: int = None):
        if 'Regressor' not in cls.__name__ and criterion == 'squared_error':
            criterion = 'gini'
        return cls.fit_(inputs, outputs, name=name, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                        n_jobs=1)

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[List[float]], predicted_inputs: List[List[float]],
                    name: str = '', n_estimators: int = 100, criterion: str = 'squared_error', max_depth: int = None):
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, n_estimators=n_estimators,
                                criterion=criterion, max_depth=max_depth, n_jobs=1)


class RandomForestRegressor(RandomForest):
    def __init__(self, n_outputs_: int, estimators_: List[List[float]] = None, name: str = ''):
        self.n_outputs_ = n_outputs_
        RandomForest.__init__(self, estimators_=estimators_, name=name)

    @classmethod
    def _skl_class(cls):
        return ensemble.RandomForestRegressor

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        model.estimators_ = [tree._instantiate_skl_model() for tree in self.estimators_]
        model.n_outputs_ = self.n_outputs_
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['estimators_'] = [DecisionTreeRegressor._instantiate_dessia_model(tree)
                                      for tree in model.estimators_]
        kwargs_dict['n_outputs_'] = model.n_outputs_
        return cls(**kwargs_dict)


class RandomForestClassifier(RandomForestRegressor):
    def __init__(self, n_classes_: int, classes_: List[int], n_outputs_: int, estimators_: List[List[float]] = None,
                 name: str = ''):
        self.n_classes_ = n_classes_
        self.classes_ = classes_
        RandomForestRegressor.__init__(self, n_outputs_=n_outputs_, estimators_=estimators_, name=name)

    @classmethod
    def _skl_class(cls):
        return ensemble.RandomForestClassifier

    def _instantiate_skl_model(self):
        model = self._call_skl_model()
        model.estimators_ = [tree._instantiate_skl_model() for tree in self.estimators_]
        model.n_classes_ = self.n_classes_
        model.n_outputs_ = self.n_outputs_
        model.classes_ = npy.array(self.classes_)
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['estimators_'] = [DecisionTreeClassifier._instantiate_dessia_model(tree)
                                      for tree in model.estimators_]
        kwargs_dict['n_classes_'] = model.n_classes_
        kwargs_dict['n_outputs_'] = model.n_outputs_
        kwargs_dict['classes_'] = model.classes_.tolist()
        return cls(**kwargs_dict)


# ======================================================================================================================
#                                                    M O D E L E R S
# ======================================================================================================================
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






