"""
Tools and base classes for machine learning methods.

"""
from typing import List, Dict, Any, Tuple
from copy import copy
import itertools

from scipy.spatial.distance import pdist, squareform
import numpy as npy
import sklearn
from sklearn import preprocessing, linear_model, ensemble, tree, svm, neural_network

try:
    from plot_data.core import Scatter, Histogram, MultiplePlots, Tooltip, ParallelPlot, PointFamily, EdgeStyle, Axis, \
        PointStyle
    from plot_data.colors import BLUE, GREY
except ImportError:
    pass
from dessia_common.core import DessiaObject
from dessia_common.utils.diff import data_eq, diff, dict_hash, list_hash
from dessia_common.utils.types import full_classname, is_sequence


# ======================================================================================================================
#                                                     S C A L E R S
# ======================================================================================================================
class BaseScaler(DessiaObject):
    _rebuild_attributes = []

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for BaseScaler. Please use children.')

    def _call_skl_scaler(self):
        return self._skl_class()()

    def instantiate_skl(self):
        scaler = self._call_skl_scaler()
        for attr in self._rebuild_attributes:
            setattr(scaler, attr, getattr(self, attr))
        return scaler

    @classmethod
    def instantiate_dessia(cls, scaler, name: str = ''):
        kwargs_dict = {'name': name}
        for attr in cls._rebuild_attributes:
            if hasattr(scaler, attr):
                if isinstance(getattr(scaler, attr), npy.ndarray):
                    kwargs_dict[attr] = getattr(scaler, attr).tolist()
                    continue
                kwargs_dict[attr] = getattr(scaler, attr)
        return cls(**kwargs_dict)

    @classmethod
    def fit(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls._skl_class()()
        scaler.fit(matrix)
        return cls.instantiate_dessia(scaler, name)

    def transform(self, matrix: List[List[float]]):
        scaler = self.instantiate_skl()
        return scaler.transform(matrix).tolist()

    @classmethod
    def fit_transform(cls, matrix: List[List[float]], name: str = ''):
        scaler = cls.fit(matrix, name)
        return scaler, scaler.transform(matrix)


class StandardScaler(BaseScaler):
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
        BaseScaler.__init__(self, name=name)

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


class LabelBinarizer(BaseScaler):
    _rebuild_attributes = ['classes_', 'y_type_', 'sparse_input_']

    def __init__(self, classes_: List[int] = None, y_type_: str = 'multiclass', sparse_input_: bool = False,
                 name: str = ''):
        self.classes_ = classes_
        self.y_type_ = y_type_
        self.sparse_input_ = sparse_input_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return preprocessing._label.LabelBinarizer

    def instantiate_skl(self):
        model = self._call_skl_scaler()
        model.classes_ = npy.array(self.classes_)
        model.y_type_ = self.y_type_
        model.sparse_input_ = self.sparse_input_
        return model


# ======================================================================================================================
#                                                        M O D E L S
# ======================================================================================================================
class BaseModel(DessiaObject):

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for BaseModel. Please use children.')

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


class LinearRegression(BaseModel):
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


class BaseTree(BaseModel):

    def __init__(self, n_classes: List[int], n_features: int, n_outputs: int, tree_state: Dict[str, Any],
                 name: str = ''):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.tree_state = tree_state
        DessiaObject.__init__(self, name=name)

    def _data_hash(self):
        hash_ = npy.linalg.norm(self.tree_state['values'][0])
        hash_ += sum(self.n_classes)
        hash_ += self.n_features
        hash_ += self.n_outputs
        return int(hash_ % 1e5)

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

    def instantiate_skl(self):
        model = self._call_skl_model()
        model = self._setstate_dessia(model, self.tree_state)
        return model

    @classmethod
    def instantiate_dessia(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['tree_state'] = cls._getstate_dessia(model)
        kwargs_dict['n_classes'] = model.n_classes.tolist()
        kwargs_dict['n_features'] = model.n_features
        kwargs_dict['n_outputs'] = model.n_outputs
        return cls(**kwargs_dict)

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[List[float]], name: str = ''):
        raise NotImplementedError('fit method is not supposed to be used in BaseTree and is not implemented.')

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[List[float]], predicted_inputs: List[List[float]],
                    name: str = ''):
        raise NotImplementedError('fit_predict method is not supposed to be used in BaseTree and is not implemented.')


class DecisionTree(BaseModel):

    def __init__(self, n_outputs_: int, tree_: BaseTree = None, name: str = ''):
        self.n_outputs_ = n_outputs_
        self.tree_ = tree_
        DessiaObject.__init__(self, name=name)

    def generic_skl_attributes(self):
        model = self._call_skl_model()
        model.n_outputs_ = self.n_outputs_
        model.tree_ = self.tree_.instantiate_skl()
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['tree_'] = BaseTree.instantiate_dessia(model.tree_)
        kwargs_dict['n_outputs_'] = model.n_outputs_
        return kwargs_dict

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

    def __init__(self, n_outputs_: int, tree_: BaseTree = None, name: str = ''):
        DecisionTree.__init__(self, n_outputs_=n_outputs_, tree_=tree_, name=name)

    @classmethod
    def _skl_class(cls):
        return tree.DecisionTreeRegressor

    def _instantiate_skl_model(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        return cls(**cls.generic_dessia_attributes(model, name=name))


class DecisionTreeClassifier(DecisionTree):
    _standalone_in_db = True

    def __init__(self, n_classes_: int, n_outputs_: int, tree_: BaseTree = None, name: str = ''):
        self.n_classes_ = n_classes_
        DecisionTree.__init__(self, n_outputs_=n_outputs_, tree_=tree_, name=name)

    @classmethod
    def _skl_class(cls):
        return tree.DecisionTreeClassifier

    def _instantiate_skl_model(self):
        model = self.generic_skl_attributes()
        model.n_classes_ = self.n_classes_
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = cls.generic_dessia_attributes(model, name=name)
        kwargs_dict['n_classes_'] = model.n_classes_
        return cls(**kwargs_dict)


class RandomForest(BaseModel):

    def __init__(self, estimators_: List[DecisionTree] = None,
                 name: str = ''):
        self.estimators_ = estimators_
        DessiaObject.__init__(self, name=name)

    # def copy(self, deep=True, memo=None):
    #     return copy(self)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for RandomForest. Please use '\
                                  'RandomForestClassifier or RandomForestRegressor.')

    def generic_skl_attributes(self):
        model = self._call_skl_model()
        model.estimators_ = [tree._instantiate_skl_model() for tree in self.estimators_]
        model.n_outputs_ = self.n_outputs_
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['n_outputs_'] = model.n_outputs_
        return kwargs_dict

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
    _standalone_in_db = True

    def __init__(self, n_outputs_: int, estimators_: List[DecisionTree] = None, name: str = ''):
        self.n_outputs_ = n_outputs_
        RandomForest.__init__(self, estimators_=estimators_, name=name)

    @classmethod
    def _skl_class(cls):
        return ensemble.RandomForestRegressor

    def _instantiate_skl_model(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = cls.generic_dessia_attributes(model, name=name)
        kwargs_dict['estimators_'] = [DecisionTreeRegressor._instantiate_dessia_model(tree)
                                      for tree in model.estimators_]
        return cls(**kwargs_dict)


class RandomForestClassifier(RandomForest):
    _standalone_in_db = True

    def __init__(self, n_classes_: int, classes_: List[int], n_outputs_: int, estimators_: List[DecisionTree] = None,
                 name: str = ''):
        self.n_outputs_ = n_outputs_
        self.n_classes_ = n_classes_
        self.classes_ = classes_
        RandomForest.__init__(self, estimators_=estimators_, name=name)

    @classmethod
    def _skl_class(cls):
        return ensemble.RandomForestClassifier

    def _instantiate_skl_model(self):
        model = self.generic_skl_attributes()
        model.n_classes_ = self.n_classes_
        model.classes_ = npy.array(self.classes_)
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = cls.generic_dessia_attributes(model, name=name)
        kwargs_dict['estimators_'] = [DecisionTreeClassifier._instantiate_dessia_model(tree)
                                      for tree in model.estimators_]
        kwargs_dict['n_classes_'] = int(model.n_classes_)
        kwargs_dict['classes_'] = model.classes_.tolist()
        return cls(**kwargs_dict)


class SVM(BaseModel):

    def __init__(self, kernel: str = 'rbf', raw_coef_: List[List[float]] = None,
                 _dual_coef_: List[List[float]] = None, _intercept_: List[float] = None, support_: List[int] = 1,
                 support_vectors_: List[List[float]] = None, _n_support: List[int] = None, _probA: List[float] = None,
                 _probB: List[float] = None, _gamma: float = 1., _sparse: bool = False, name: str = ''):
        self.kernel = kernel
        self.raw_coef_ = raw_coef_
        self._dual_coef_ = _dual_coef_
        self._intercept_ = _intercept_
        self.support_ = support_
        self.support_vectors_ = support_vectors_
        self._n_support = _n_support
        self._probA = _probA
        self._probB = _probB
        self._gamma = _gamma
        self._sparse = _sparse
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for SVM. Please use SVC or SVR.')

    def _call_skl_model(self):
        return self._skl_class()(kernel=self.kernel)

    def generic_skl_attributes(self):
        model = self._call_skl_model()
        model.raw_coef_ = npy.array(self.raw_coef_)
        model._dual_coef_ = npy.array(self._dual_coef_)
        model._intercept_ = npy.array(self._intercept_)
        model.support_ = npy.array(self.support_, dtype=npy.int32)
        model.support_vectors_ = npy.array(self.support_vectors_)
        model._n_support = npy.array(self._n_support, dtype=npy.int32)
        model._probA = npy.array(self._probA)
        model._probB = npy.array(self._probB)
        model._gamma = self._gamma
        model._sparse = self._sparse
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['kernel'] = model.kernel
        kwargs_dict['raw_coef_'] = model._get_coef().tolist()
        kwargs_dict['_dual_coef_'] = model._dual_coef_.tolist()
        kwargs_dict['_intercept_'] = model._intercept_.tolist()
        kwargs_dict['support_'] = model.support_.tolist()
        kwargs_dict['support_vectors_'] = model.support_vectors_.tolist()
        kwargs_dict['_n_support'] = model._n_support.tolist()
        kwargs_dict['_probA'] = model._probA.tolist()
        kwargs_dict['_probB'] = model._probB.tolist()
        kwargs_dict['_gamma'] = float(model._gamma)
        kwargs_dict['_sparse'] = model._sparse
        return kwargs_dict

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[float], C: float = 1., kernel: str = 'rbf', name: str = ''):
        return cls.fit_(inputs, outputs, name=name, C=C, kernel=kernel)

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[float], predicted_inputs: List[List[float]],
                    C: float = 1., kernel: str = 'rbf', name: str = ''):
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, C=C, kernel=kernel)


class SVR(SVM):
    _standalone_in_db = True

    def __init__(self, kernel: str = 'rbf', raw_coef_: List[List[float]] = None,
                 _dual_coef_: List[List[float]] = None, _intercept_: List[float] = None, support_: List[int] = 1,
                 support_vectors_: List[List[float]] = None, _n_support: List[int] = None, _probA: List[float] = None,
                 _probB: List[float] = None, _gamma: float = 1., _sparse: bool = False, name: str = ''):
        SVM.__init__(self, raw_coef_=raw_coef_, _dual_coef_=_dual_coef_, support_vectors_=support_vectors_,
                     _sparse=_sparse, kernel=kernel, _n_support=_n_support, support_=support_, _intercept_=_intercept_,
                     _probA=_probA, _probB=_probB, _gamma=_gamma, name=name)

    @classmethod
    def _skl_class(cls):
        return svm.SVR

    def _instantiate_skl_model(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        return cls(**cls.generic_dessia_attributes(model, name=name))


class SVC(SVM):
    _standalone_in_db = True

    def __init__(self, kernel: str = 'rbf', raw_coef_: List[List[float]] = None,
                 _dual_coef_: List[List[float]] = None, _intercept_: List[float] = None, support_: List[int] = 1,
                 support_vectors_: List[List[float]] = None, _n_support: List[int] = None, _probA: List[float] = None,
                 _probB: List[float] = None, _gamma: float = 1., _sparse: bool = False,
                 classes_: List[int] = None, name: str = ''):
        self.classes_ = classes_
        SVM.__init__(self, raw_coef_=raw_coef_, _dual_coef_=_dual_coef_, support_vectors_=support_vectors_,
                    _sparse=_sparse, kernel=kernel, _n_support=_n_support, support_=support_, _intercept_=_intercept_,
                    _probA=_probA, _probB=_probB, _gamma=_gamma, name=name)

    @classmethod
    def _skl_class(cls):
        return svm.SVC

    def _instantiate_skl_model(self):
        model = self.generic_skl_attributes()
        model.classes_ = npy.array(self.classes_)
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = cls.generic_dessia_attributes(model, name=name)
        kwargs_dict['classes_'] = model.classes_.tolist()
        return cls(**kwargs_dict)


class MLP(BaseModel):

    def __init__(self, coefs_: List[List[List[float]]] = None, intercepts_: List[List[float]] = None,
                 n_layers_: int = None, activation: str = 'relu', out_activation_: str = 'identity', name: str = ''):
        self.coefs_ = coefs_
        self.intercepts_ = intercepts_
        self.n_layers_ = n_layers_
        self.activation = activation
        self.out_activation_ = out_activation_
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for MLP. Please use MLPRegressor or '\
                                  'MLPClassifier.')

    def _call_skl_model(self):
        return self._skl_class()()

    def generic_skl_attributes(self):
        model = self._call_skl_model()
        model.coefs_ = [npy.array(coefs_) for coefs_ in self.coefs_]
        model.intercepts_ = [npy.array(intercepts_) for intercepts_ in self.intercepts_]
        model.n_layers_ = self.n_layers_
        model.activation = self.activation
        model.out_activation_ = self.out_activation_
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        kwargs_dict = {'name': name}
        kwargs_dict['coefs_'] = [coefs_.tolist() for coefs_ in model.coefs_]
        kwargs_dict['intercepts_'] = [intercepts_.tolist() for intercepts_ in model.intercepts_]
        kwargs_dict['n_layers_'] = model.n_layers_
        kwargs_dict['activation'] = model.activation
        kwargs_dict['out_activation_'] = model.out_activation_
        return kwargs_dict

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[float], hidden_layer_sizes: List[int] = None,
            activation: str = 'relu', alpha: float = 0.0001, solver: str = 'adam', max_iter: int = 200,
            tol: float = 0.0001, name: str = ''):
        return cls.fit_(inputs, outputs, name=name, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                        alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)

    @classmethod
    def fit_predict(cls, inputs: List[List[float]], outputs: List[float], predicted_inputs: List[List[float]],
                    hidden_layer_sizes: List[int] = None, activation: str = 'relu', alpha: float = 0.0001,
                    solver: str = 'adam', max_iter: int = 200, tol: float = 0.0001, name: str = ''):
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation, alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)


class MLPRegressor(MLP):
    _standalone_in_db = True

    def __init__(self, coefs_: List[List[List[float]]] = None, intercepts_: List[List[float]] = None,
                 n_layers_: int = None, activation: str = 'relu', out_activation_: str = 'identity', name: str = ''):
        MLP.__init__(self, coefs_=coefs_, intercepts_=intercepts_, n_layers_=n_layers_, activation=activation,
                     out_activation_=out_activation_, name=name)

    @classmethod
    def _skl_class(cls):
        return neural_network.MLPRegressor

    def _instantiate_skl_model(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        return cls(**cls.generic_dessia_attributes(model, name=name))


class MLPClassifier(MLP):
    _standalone_in_db = True

    def __init__(self, coefs_: List[List[List[float]]] = None, intercepts_: List[List[float]] = None,
                 n_layers_: int = None, activation: str = 'relu', out_activation_: str = 'identity',
                 n_outputs_: int = None, _label_binarizer: LabelBinarizer = None,name: str = ''):
        self.n_outputs_ = n_outputs_
        self._label_binarizer = _label_binarizer
        MLP.__init__(self, coefs_=coefs_, intercepts_=intercepts_, n_layers_=n_layers_, activation=activation,
                     out_activation_=out_activation_, name=name)

    @classmethod
    def _skl_class(cls):
        return neural_network.MLPClassifier

    def _instantiate_skl_model(self):
        model = self.generic_skl_attributes()
        model.n_outputs_ = self.n_outputs_
        model._label_binarizer = self._label_binarizer.instantiate_skl()
        return model

    @classmethod
    def _instantiate_dessia_model(cls, model, name: str = ''):
        kwargs_dict = cls.generic_dessia_attributes(model, name=name)
        kwargs_dict['n_outputs_'] = model.n_outputs_
        kwargs_dict['_label_binarizer'] = LabelBinarizer.instantiate_dessia(model._label_binarizer)
        return cls(**kwargs_dict)


# ======================================================================================================================
#                                                    M O D E L E R S
# ======================================================================================================================
class Modeler(DessiaObject):
    def __init__(self, model: BaseModel, scaler: BaseScaler, scaled_inputs: bool = True, scaled_outputs: bool = False,
                 name: str = ''):
        self.model = model
        self.scaler = scaler
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






