"""
Tools and base classes for machine learning methods.
"""
from typing import List, Dict, Any, Tuple, Union
import random

import numpy as npy
from sklearn import preprocessing, linear_model, ensemble, tree, svm, neural_network

from dessia_common.core import DessiaObject

Vector = List[float]
Matrix = List[Vector]

# ======================================================================================================================
#                                                     S C A L E R S
# ======================================================================================================================
class Scaler(DessiaObject):
    """Base object for handling a scikit-learn Scaler."""
    _rebuild_attributes = []

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for Scaler. Please use children.')

    def _init_empty(self):
        return self._skl_class()()

    @staticmethod
    def _set_class(is_scaled: bool) -> 'Scaler':
        if is_scaled:
            return StandardScaler
        return IdentityScaler

    @staticmethod
    def _set_name(modeler_name: str, in_out: str, is_scaled: bool) -> str:
        name = f"{modeler_name}_"
        return name + (f"{in_out}_scaler" if is_scaled else "indentity_scaler")

    @classmethod
    def set_in_modeler(cls, modeler_name: str, in_out: str, is_scaled: bool) -> 'Scaler':
        class_ = cls._set_class(is_scaled)
        name = cls._set_name(modeler_name, in_out, is_scaled)
        return class_, name

    def instantiate_skl(self):
        """Instantiate scikit-learn Scaler from Scaler object, or children."""
        scaler = self._init_empty()
        for attr in self._rebuild_attributes:
            setattr(scaler, attr, getattr(self, attr))
        return scaler

    @classmethod
    def instantiate_dessia(cls, scaler, name: str = '') -> 'Scaler':
        """Instantiate Scaler object, or children, from scikit-learn scaler."""
        kwargs = {attr: get_scaler_attr(scaler, attr) for attr in cls._rebuild_attributes}
        kwargs["name"] = name
        return cls(**kwargs)

    @classmethod
    def fit(cls, matrix: Matrix, name: str = '') -> 'Scaler':
        """
        Fit scaler with data stored in matrix.

        :param matrix:
            Matrix of data of dimension `n_samples x n_features`
        :type matrix: Matrix

        :param name:
            Name of Scaler
        :type name: str, `optional`, defaults to `''`

        :return: The Scaler or children (DessiaObject) fit on matrix.
        :rtype: Scaler
        """
        scaler = cls._skl_class()()
        reshaped_matrix = vector_to_2d_matrix(matrix)
        scaler.fit(reshaped_matrix)
        return cls.instantiate_dessia(scaler, name)

    def transform(self, matrix: Matrix) -> Matrix:
        """
        Transform the data stored in matrix according to this Scaler or children.

        :param matrix:
            Matrix of data of dimension `n_samples x n_features`
        :type matrix: List[List[float]]

        :return: The scaled matrix according to the rules set of scaler.
        :rtype: List[List[float]]
        """
        scaler = self.instantiate_skl()
        reshaped_matrix = vector_to_2d_matrix(matrix)
        return scaler.transform(reshaped_matrix).tolist()


    def inverse_transform(self, matrix: Matrix) -> Matrix:
        """
        Inverse transform the scaled data stored in matrix according to this Scaler or children.

        :param matrix:
            Scaled matrix of data of dimension `n_samples x n_features`
        :type matrix: List[List[float]]

        :return: The raw matrix according to the rules of scaler.
        :rtype: List[List[float]]
        """
        scaler = self.instantiate_skl()
        reshaped_matrix = vector_to_2d_matrix(matrix)
        return scaler.inverse_transform(reshaped_matrix).tolist()

    @classmethod
    def fit_transform(cls, matrix: Matrix, name: str = '') -> Tuple['Scaler', Matrix]:
        """
        Fit scaler with data stored in matrix and transform it. It is the succession of fit and transform methods.
        """
        reshaped_matrix = vector_to_2d_matrix(matrix)
        scaler = cls.fit(reshaped_matrix, name)
        return scaler, scaler.transform(reshaped_matrix)

    def transform_matrices(self, *matrices: Tuple[Matrix]) -> Tuple[Matrix]:
        scaled_matrices = tuple()
        for matrix in matrices:
            scaled_matrices += (self.transform(matrix), )
        return scaled_matrices

    def inverse_transform_matrices(self, *scaled_matrices: Tuple[Matrix]) -> Tuple[Matrix]:
        unscaled_matrices = tuple()
        for matrix in scaled_matrices:
            unscaled_matrices += (self.inverse_transform(matrix), )
        return unscaled_matrices


class StandardScaler(Scaler):
    """
    Data scaler that standardly scale data. The operation made by this scaler is `new_X = (X - mean(X))/std(X)`.

    :param mean_:
        List of means
    :type mean_: List[float], `optional`, defaults to `None`

    :param scale_:
        List of standard deviations
    :type scale_: List[float], `optional`, defaults to `None`

    :param var_:
        List of variances
    :type var_: List[float], `optional`, defaults to `None`
    """
    _rebuild_attributes = ['mean_', 'scale_', 'var_']
    _standalone_in_db = True

    def __init__(self, mean_: Vector = None, scale_: Vector = None, var_: Vector = None, name: str = ''):
        self.mean_ = mean_
        self.scale_ = scale_
        self.var_ = var_
        Scaler.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return preprocessing.StandardScaler


class IdentityScaler(StandardScaler):
    """Data scaler that scales nothing."""

    def __init__(self, mean_: Vector = None, scale_: Vector = None, var_: Vector = None, name: str = 'identity_scaler'):
        StandardScaler.__init__(self, mean_=mean_, scale_=scale_, var_=var_, name=name)

    def _init_empty(self):
        return self._skl_class()(with_mean = False, with_std = False)


class LabelBinarizer(Scaler):
    """
    Data scaler used in MLPClassifier to standardize class labels. Only implemented for MLPClassifier to work correctly.

    :param classes_:
        List of classes to standardize. Can be any int.
    :type classes_: List[int]

    :param y_type_:
        Type of output labels.
    :type y_type_: str

    :param sparse_input_:
        Specify if the inputs are a sparse matrix or not.
    :type sparse_input_: bool, defaults to `False`
    """
    _rebuild_attributes = ['classes_', 'y_type_', 'sparse_input_']

    def __init__(self, classes_: List[int] = None, y_type_: str = 'multiclass', sparse_input_: bool = False,
                 name: str = ''):
        self.classes_ = classes_
        self.y_type_ = y_type_
        self.sparse_input_ = sparse_input_
        Scaler.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return preprocessing._label.LabelBinarizer

    def instantiate_skl(self):
        """Instantiate scikit-learn LabelBinarizer from LabelBinarizer object."""
        scaler = self._init_empty()
        scaler.classes_ = npy.array(self.classes_)
        scaler.y_type_ = self.y_type_
        scaler.sparse_input_ = self.sparse_input_
        return scaler


# ======================================================================================================================
#                                                        M O D E L S
# ======================================================================================================================
class Model(DessiaObject):
    """Base object for handling a scikit-learn models (classifier and regressor)."""

    def __init__(self, name: str = ''):
        DessiaObject.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError(f'Method _skl_class not implemented for {cls.__name__}.')

    def _call_skl_model(self):
        return self._skl_class()()

    def _instantiate_skl(self):
        raise NotImplementedError(f'Method _instantiate_skl not implemented for {type(self).__name__}.')

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        raise NotImplementedError(f'Method _instantiate_dessia not implemented for {cls.__name__}.')

    @classmethod
    def init_for_modeler_(cls, **hyperparameters: Dict[str, Any]) -> Tuple['Model', Dict[str, Any], str]:
        """
        Initialize class of Model with its name and hyperparemeters to fit in Modeler.

        :return: The Model class, the hyperparameters to instantiate it and the future name of instance.
        :rtype: Tuple['Model', Dict[str, Any], str]
        """
        return cls, hyperparameters

    @classmethod
    def fit_(cls, inputs: Matrix, outputs: Matrix, name: str = '', **hyperparameters) -> 'Model':
        """
        Standard method to fit outputs to inputs thanks to a scikit-learn model.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param name:
            Name of Model
        :type name: str, `optional`, defaults to `''`

        :param hyperparameters:
            Hyperparameters of the used scikit-learn object.
        :type hyperparameters: dict[str, Any], `optional`

        :return: The Model or children (DessiaObject) fit on matrix.
        :rtype: Model
        """
        model = cls._skl_class()(**hyperparameters)
        model.fit(inputs, outputs)
        return cls._instantiate_dessia(model, name)

    def predict(self, inputs: Matrix) -> Union[Vector, Matrix]:
        """
        Standard method to predict outputs from inputs with a Model or children.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :return: The predicted values for inputs.
        :rtype: Union[List[float], List[List[float]]]
        """
        model = self._instantiate_skl()
        return model.predict(inputs).tolist()

    @classmethod
    def fit_predict_(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, name: str = '',
                     **hyperparameters) -> Tuple['Model', Union[Vector, Matrix]]:
        """
        Fit outputs to inputs and predict outputs for predicted_inputs. It is the succession of fit and predict methods.
        """
        model = cls.fit_(inputs, outputs, name, **hyperparameters)
        return model, model.predict(predicted_inputs)

    def score(self, inputs: Matrix, outputs: Matrix) -> float:
        """
        Compute the score of Model or children.

        Please be sure to fit the model before computing its score and use test data and not train data.
        Train data is data used to train the model and shall not be used to evaluate its quality.
        Test data is data used to test the model and must not be used to train (fit) it.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :return: The score of Model or children (DessiaObject).
        :rtype: float
        """
        model = self._instantiate_skl()
        return model.score(inputs, outputs)


class LinearModel(Model):
    """
    Abstract class for linear models.

    :param coef_:
        List of coefficients of the model. Each element (i, j) of coef_ is the slope of the linear model predicting
        the i-th output from the j-th input.
    :type coef_: List[List[float]]

    :param intercept_:
        List of offsets of the model. Each element (i, ) of intercept_ is added to the prediction made with coef_ to
        compute the i-th element of outputs prediction.
    :type intercept_: List[float]

    :param name:
        Name of LinearModel regression
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, coef_: Matrix = None, intercept_: Matrix = None, name: str = ''):
        self.coef_ = coef_
        self.intercept_ = intercept_
        Model.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for LinearModel. Please use '\
                                  'Ridge or LinearRegression.')

    def _instantiate_skl(self):
        model = self._call_skl_model()
        model.coef_ = npy.array(self.coef_)
        model.intercept_  = npy.array(self.intercept_)
        return model

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        return cls(coef_=model.coef_.tolist(), intercept_=model.intercept_.tolist(), name=name)


class Ridge(LinearModel):
    """
    Ridge regression. It is a linear or least square regression but computed with a regularization term `alpha`.

    The model searched with this method is of the form `Y = A.X + B`, where `Y` are the ouputs, `X` the inputs, `A` and
    `B` the matrices of the model.

    The function minimized to get the linear model is `|| Y - A.X + B || + alpha.|| A || = 0`. This means setting
    `alpha` to `0` is equivalent than searching a linear model from a least square regression.

    More information: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    :param coef_:
        List of coefficients of the model. Each element (i, j) of coef_ is the slope of the linear model predicting
        the i-th output from the j-th input.
    :type coef_: List[List[float]]

    :param intercept_:
        List of offsets of the model. Each element (i, ) of intercept_ is added to the prediction made with coef_ to
        compute the i-th element of outputs prediction.
    :type intercept_: List[float]

    :param name:
        Name of Ridge regression
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, coef_: Matrix = None, intercept_: Matrix = None, name: str = ''):
        LinearModel.__init__(self, coef_=coef_, intercept_=intercept_, name=name)

    @classmethod
    def _skl_class(cls):
        return linear_model.Ridge

    @classmethod
    def init_for_modeler(cls, alpha: float = 1., fit_intercept: bool = True,
                         tol: float = 0.001) -> Tuple['Ridge', Dict[str, Any], str]:
        """
        Initialize class Ridge with its name and hyperparemeters to fit in Modeler.

        :param alpha:
            Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative
            float i.e. in [0, inf[. When alpha = 0, the objective is equivalent to ordinary least squares,
            solved by the LinearRegression object. For numerical reasons, using `alpha = 0` with the Ridge object is
            not advised. Instead, you should use the LinearRegression object. If an array is passed, penalties are
            assumed to be specific to the targets. Hence they must correspond in number.
        :type alpha: float, `optional`, defaults to 1.

        :param fit_intercept:
            Whether to fit the intercept for this model. If set to False, no intercept will be used in calculations
            (i.e. X and Y are expected to be centered).
        :type fit_intercept: bool, `optional`, defaults to True

        :param tol:
            Precision of the solution.
        :type tol: float, `optional`, defaults to 0.001

        :return: The Ridge class, the hyperparameters to instantiate it and the future name of instance.
        :rtype: Tuple['Ridge', Dict[str, Any], str]
        """
        return cls.init_for_modeler_(alpha=alpha, fit_intercept=fit_intercept, tol=tol)

    @classmethod
    def fit(cls, inputs: Matrix, outputs: Matrix, alpha: float = 1., fit_intercept: bool = True, tol: float = 0.001,
            name: str = '') -> 'Ridge':
        """
        Standard method to fit outputs to inputs thanks to Ridge linear model from scikit-learn.

        More information: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param alpha:
            Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative
            float i.e. in [0, inf[. When alpha = 0, the objective is equivalent to ordinary least squares,
            solved by the LinearRegression object. For numerical reasons, using `alpha = 0` with the Ridge object is
            not advised. Instead, you should use the LinearRegression object. If an array is passed, penalties are
            assumed to be specific to the targets. Hence they must correspond in number.
        :type alpha: float, `optional`, defaults to 1.

        :param fit_intercept:
            Whether to fit the intercept for this model. If set to False, no intercept will be used in calculations
            (i.e. X and Y are expected to be centered).
        :type fit_intercept: bool, `optional`, defaults to True

        :param tol:
            Precision of the solution.
        :type tol: float, `optional`, defaults to 0.001

        :param name:
            Name of Ridge model
        :type name: str, `optional`, defaults to `''`

        :return: The Ridge model fit on inputs and outputs.
        :rtype: Ridge
        """
        return cls.fit_(inputs, outputs, name=name, alpha=alpha, fit_intercept=fit_intercept, tol=tol)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, alpha: float = 1.,
                    fit_intercept: bool = True, tol: float = 0.001,
                    name: str = '') -> Tuple['Ridge', Union[Vector, Matrix]]:
        """
        Fit outputs to inputs and predict outputs for predicted_inputs. It is the succession of fit and predict methods.
        """
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name,
                                alpha=alpha, fit_intercept=fit_intercept, tol=tol)


class LinearRegression(LinearModel):
    """
    Linear regression.

    The model searched with this method is of the form `Y = A.X + B`, where `Y` are the ouputs, `X` the inputs, `A` and
    `B` the matrices of the model.

    The function minimized to get the linear model is `|| Y - A.X + B || = 0`.

    More information: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    :param coef_:
        List of coefficients of the model. Each element (i, j) of coef_ is the slope of the linear model predicting
        the i-th output from the j-th input.
    :type coef_: List[List[float]]

    :param intercept_:
        List of offsets of the model. Each element (i, ) of intercept_ is added to the prediction made with coef_ to
        compute the i-th element of outputs prediction.
    :type intercept_: List[float]

    :param name:
        Name of Linear regression
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, coef_: Matrix = None, intercept_: Matrix = None, name: str = ''):
        LinearModel.__init__(self, coef_=coef_, intercept_=intercept_, name=name)

    @classmethod
    def _skl_class(cls):
        return linear_model.LinearRegression

    @classmethod
    def init_for_modeler(cls, fit_intercept: bool = True,
                         positive: bool = False) -> Tuple['LinearRegression', Dict[str, Any], str]:
        """
        Initialize class LinearRegression with its name and hyperparemeters to fit in Modeler.

        :param fit_intercept:
            Whether to fit the intercept for this model. If set to False, no intercept will be used in calculations
            (i.e. X and Y are expected to be centered).
        :type fit_intercept: bool, `optional`, defaults to True

        :param positive:
            When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
        :type positive: bool, `optional`, defaults to False

        :return: The LinearRegression model fit on inputs and outputs.
        :rtype: Tuple['LinearRegression', Dict[str, Any], str]
        """
        return cls.init_for_modeler_(fit_intercept=fit_intercept, positive=positive)

    @classmethod
    def fit(cls, inputs: Matrix, outputs: Matrix, fit_intercept: bool = True, positive: bool = False,
            name: str = '') -> 'LinearRegression':
        """
        Standard method to fit outputs to inputs thanks to Linear Regression model from scikit-learn.

        More information: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param fit_intercept:
            Whether to fit the intercept for this model. If set to False, no intercept will be used in calculations
            (i.e. X and Y are expected to be centered).
        :type fit_intercept: bool, `optional`, defaults to True

        :param positive:
            When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.
        :type positive: bool, `optional`, defaults to False

        :param name:
            Name of LinearRegression model
        :type name: str, `optional`, defaults to `''`

        :return: The Linear model fit on inputs and outputs.
        :rtype: LinearRegression
        """
        return cls.fit_(inputs, outputs, name=name, fit_intercept=fit_intercept, positive=positive)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, fit_intercept: bool = True,
                    positive: bool = False, name: str = '') -> Tuple['LinearRegression', Union[Vector, Matrix]]:
        """
        Fit outputs to inputs and predict outputs for predicted_inputs. It is the succession of fit and predict methods.
        """
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name,
                                fit_intercept=fit_intercept, positive=positive)


class Tree(Model):
    """
    Base object for handling a scikit-learn tree._tree.Tree object (Cython).

    Please refer to https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html for more
    information on attributes of Tree object and understanding the decision tree structure for basic usage.

    :param n_classes:
        Number of output classes to predict from data.
    :type n_classes: List[int], `optional`, defaults to `None`

    :param n_features:
        Number of features to handle in data.
    :type n_features: int, `optional`, defaults to `None`

    :param n_outputs:
        The number of outputs when fit is performed.
    :type n_outputs: int, `optional`, defaults to `None`

    :param tree_state:
        All required values to re-instantiate a fully working scikit-learn tree are stored in this parameter.
    :type tree_state: Dict[str, Any], `optional`, defaults to `None`

    :param name:
        Name of Tree
    :type name: str, `optional`, defaults to `''`
    """

    def __init__(self, n_classes: List[int] = None, n_features: int = None, n_outputs: int = None,
                 tree_state: Dict[str, Any] = None, name: str = ''):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.tree_state = tree_state
        Model.__init__(self, name=name)

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
        dessia_state = {'max_depth': int(state['max_depth']),
                        'node_count': int(state['node_count']),
                        'values': state['values'].tolist(),
                        'nodes': {'dtypes': state['nodes'].dtype.descr, 'values': state['nodes'].tolist()}}
        return dessia_state

    @staticmethod
    def _setstate_dessia(model, state):
        skl_state = {'max_depth': int(state['max_depth']),
                     'node_count': int(state['node_count']),
                     'values': npy.array(state['values']),
                     'nodes': npy.array(state['nodes']['values'], dtype=state['nodes']['dtypes'])}
        model.__setstate__(skl_state)
        return model

    def _instantiate_skl(self):
        model = self._call_skl_model()
        model = self._setstate_dessia(model, self.tree_state)
        return model

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        kwargs = {'name': name,
                  'tree_state': cls._getstate_dessia(model),
                  'n_classes': model.n_classes.tolist(),
                  'n_features': model.n_features,
                  'n_outputs': model.n_outputs}
        return cls(**kwargs)


class DecisionTreeRegressor(Model):
    """
    Base class for handling scikit-learn DecisionTreeRegressor.

    More information: https://scikit-learn.org/stable/modules/tree.html#tree

    :param n_outputs_:
        The number of outputs when fit is performed.
    :type n_outputs_: int, `optional`, defaults to `None`

    :param tree_:
        The underlying Tree object.
        Please refer to https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html for
        attributes of Tree object and understanding the decision tree structure for basic usage of these attributes.
    :type tree_: Tree, `optional`, defaults to `None`

    :param name:
        Name of DecisionTreeRegressor
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, n_outputs_: int = None, tree_: Tree = None, name: str = ''):
        self.n_outputs_ = n_outputs_
        self.tree_ = tree_
        Model.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        return tree.DecisionTreeRegressor

    def generic_skl_attributes(self):
        """Generic method (shared between trees) to set scikit-learn model attributes from self attributes."""
        model = self._call_skl_model()
        model.n_outputs_ = self.n_outputs_
        model.tree_ = self.tree_._instantiate_skl()
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        """Generic method (shared between trees) to set self attributes from scikit-learn model attributes."""
        return {'name': name,
                'tree_': Tree._instantiate_dessia(model.tree_),
                'n_outputs_': model.n_outputs_}

    def _instantiate_skl(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        return cls(**cls.generic_dessia_attributes(model, name=name))

    @classmethod
    def _check_criterion(cls, criterion: str):
        if 'egressor' not in cls.__name__ and criterion == 'squared_error':
            return 'gini'
        return criterion

    @classmethod
    def init_for_modeler(cls, criterion: str = 'squared_error',
                         max_depth: int = None) -> Tuple['DecisionTreeRegressor', Dict[str, Any], str]:
        """
        Initialize class DecisionTreeRegressor with its name and hyperparemeters to fit in Modeler.

        :param criterion:
            The function to measure the quality of a split. Supported criteria are “squared_error” for the mean
            squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2
            loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s
            improvement score for potential splits, “absolute_error” for the mean absolute error, which minimizes the
            L1 loss using the median of each terminal node, and “poisson” which uses reduction in Poisson deviance to
            find splits.
        :type criterion: str, `optional`, defaults to 'squared_error'

        :param max_depth:
            The maximum depth of the tree. If `None`, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.
        :type max_depth: int, `optional`, defaults to `None`

        :return: The DecisionTreeRegressor model fit on inputs and outputs.
        :rtype: Tuple['DecisionTreeRegressor', Dict[str, Any], str]
        """
        return cls.init_for_modeler_(criterion=criterion, max_depth=max_depth)

    @classmethod
    def fit(cls, inputs: Matrix, outputs: Matrix, criterion: str = 'squared_error', max_depth: int = None,
            name: str = '') -> 'DecisionTreeRegressor':
        """
        Standard method to fit outputs to inputs thanks to DecisionTreeRegressor model from scikit-learn.

        More information: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param criterion:
            The function to measure the quality of a split. Supported criteria are “squared_error” for the mean
            squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2
            loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with Friedman’s
            improvement score for potential splits, “absolute_error” for the mean absolute error, which minimizes the
            L1 loss using the median of each terminal node, and “poisson” which uses reduction in Poisson deviance to
            find splits.
        :type criterion: str, `optional`, defaults to 'squared_error'

        :param max_depth:
            The maximum depth of the tree. If `None`, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.
        :type max_depth: int, `optional`, defaults to `None`

        :param name:
            Name of DecisionTreeRegressor model
        :type name: str, `optional`, defaults to `''`

        :return: The DecisionTreeRegressor model fit on inputs and outputs.
        :rtype: DecisionTreeRegressor
        """
        criterion = cls._check_criterion(criterion)
        return cls.fit_(inputs, outputs, name=name, criterion=criterion, max_depth=max_depth)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, criterion: str = 'squared_error',
                    max_depth: int = None, name: str = '') -> Tuple['DecisionTreeRegressor', Union[Vector, Matrix]]:
        """
        Succession of fit and predict methods: fit outputs to inputs and predict outputs for predicted_inputs.
        """
        criterion = cls._check_criterion(criterion)
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, criterion=criterion, max_depth=max_depth)


class DecisionTreeClassifier(DecisionTreeRegressor):
    """
    Base class for handling scikit-learn DecisionTreeClassifier.

    More information: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    :param n_classes_:
        The number of classes (for single output problems), or a list containing the number of classes for each output
        (for multi-output problems).
    :type n_classes_: Union[int, List[int]], `optional`, defaults to `None`

    :param classes_:
        The number of outputs when fit is performed.
    :type classes_: List[int], `optional`, defaults to `None`

    :param n_outputs_:
        The number of outputs when fit is performed.
    :type n_outputs_: int, `optional`, defaults to `None`

    :param tree_:
        The underlying Tree object.
        Please refer to https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html for
        attributes of Tree object and understanding the decision tree structure for basic usage of these attributes.
    :type tree_: Tree, `optional`, defaults to `None`

    :param name:
        Name of DecisionTreeClassifier
    :type name: str, `optional`, defaults to `''`
    """

    def __init__(self, n_classes_: Union[int, List[int]] = None, classes_: List[int] = None, n_outputs_: int = None,
                 tree_: Tree = None, name: str = ''):
        self.n_classes_ = n_classes_
        self.classes_ = classes_
        DecisionTreeRegressor.__init__(self, n_outputs_=n_outputs_, tree_=tree_, name=name)

    @classmethod
    def _skl_class(cls):
        return tree.DecisionTreeClassifier

    def _instantiate_skl(self):
        model = self.generic_skl_attributes()
        model.n_classes_ = self.n_classes_
        model.classes_ = npy.array(self.classes_)
        return model

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        kwargs = cls.generic_dessia_attributes(model, name=name)
        kwargs.update({'n_classes_': (model.n_classes_ if isinstance(model.n_classes_, (int, list))
                                      else model.n_classes_.tolist()),
                       'classes_': (model.classes_.tolist() if isinstance(model.classes_, npy.ndarray)
                                    else [klass.tolist() for klass in model.classes_])})
        return cls(**kwargs)

    def score(self, inputs: Matrix, outputs: Matrix) -> float:
        """
        Compute the score of Model or children.

        Please be sure to fit the model before computing its score and use test data and not train data.
        Train data is data used to train the model and shall not be used to evaluate its quality.
        Test data is data used to test the model and must not be used to train (fit) it.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :return: The score of Model or children (DessiaObject).
        :rtype: float
        """
        model = self._instantiate_skl()
        if self.n_outputs_==1:
            return model.score(inputs, outputs)
        raise ValueError('multiclass-multioutput is not supported')


class RandomForest(Model):
    """
    Base object for handling a scikit-learn RandomForest object.

    Please refer to https://scikit-learn.org/stable/modules/ensemble.html#forest for more information on RandomForest.

    :param n_outputs_:
        The number of outputs when fit is performed.
    :type n_outputs_: int, `optional`, defaults to `None`

    :param estimators_:
        The collection of fitted sub-trees.
    :type estimators_: List[DecisionTreeRegressor], `optional`, defaults to `None`

    :param name:
        Name of RandomForest
    :type name: str, `optional`, defaults to `''`
    """

    def __init__(self, n_outputs_: int = None, estimators_: List[DecisionTreeRegressor] = None, name: str = ''):
        self.estimators_ = estimators_
        self.n_outputs_ = n_outputs_
        Model.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for RandomForest. Please use '\
                                  'RandomForestClassifier or RandomForestRegressor.')

    def generic_skl_attributes(self):
        """
        Generic method (shared between RandomForest) to set scikit-learn model attributes from self attributes.
        """
        model = self._call_skl_model()
        model.estimators_ = [tree._instantiate_skl() for tree in self.estimators_]
        model.n_outputs_ = self.n_outputs_
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        """
        Generic method (shared between RandomForest) to set self attributes from scikit-learn model attributes.
        """
        return {'name': name,
                'n_outputs_': model.n_outputs_}

    @classmethod
    def _check_criterion(cls, criterion: str):
        if 'egressor' not in cls.__name__ and criterion == 'squared_error':
            return 'gini'
        return criterion

    @classmethod
    def init_for_modeler(cls, n_estimators: int = 100, criterion: str = 'squared_error',
                         max_depth: int = None) -> Tuple['RandomForest', Dict[str, Any], str]:
        """
        Initialize class RandomForest with its name and hyperparemeters to fit in Modeler.

        :param n_estimators:
            Number of DecisionTree contained in RandomForestRegressor or RandomForestClassifier
        :type n_estimators: int, `optional`, defaults to 100

        :param criterion:
         |  - **Regressor:** The function to measure the quality of a split. Supported criteria are “squared_error” for
            the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes
            the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with
            Friedman’s improvement score for potential splits, “absolute_error” for the mean absolute error,
            which minimizes the L1 loss using the median of each terminal node, and “poisson” which uses reduction in
            Poisson deviance to find splits.

         |  - **Classifier:** The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation.
         |  Note: This parameter is tree-specific.
        :type criterion: str, `optional`, defaults to 'squared_error'

        :param max_depth:
            The maximum depth of the tree. If `None`, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.
        :type max_depth: int, `optional`, defaults to `None`

        :return: The RandomForest model fit on inputs and outputs.
        :rtype: Tuple['RandomForest', Dict[str, Any], str]
        """
        return cls.init_for_modeler_(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    @classmethod
    def fit(cls, inputs: Matrix, outputs: Matrix, n_estimators: int = 100, criterion: str = 'squared_error',
            max_depth: int = None, name: str = '') -> 'RandomForest':
        """
        Standard method to fit outputs to inputs thanks to RandomForest model from scikit-learn.

        More information:
            - Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            - Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param n_estimators:
            Number of DecisionTree contained in RandomForestRegressor or RandomForestClassifier
        :type n_estimators: int, `optional`, defaults to 100

        :param criterion:
         |  - **Regressor:** The function to measure the quality of a split. Supported criteria are “squared_error” for
            the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes
            the L2 loss using the mean of each terminal node, “friedman_mse”, which uses mean squared error with
            Friedman’s improvement score for potential splits, “absolute_error” for the mean absolute error,
            which minimizes the L1 loss using the median of each terminal node, and “poisson” which uses reduction in
            Poisson deviance to find splits.

         |  - **Classifier:** The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation.
         |  Note: This parameter is tree-specific.
        :type criterion: str, `optional`, defaults to 'squared_error'

        :param max_depth:
            The maximum depth of the tree. If `None`, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.
        :type max_depth: int, `optional`, defaults to `None`

        :param name:
            Name of RandomForestRegressor or RandomForestClassifier model
        :type name: str, `optional`, defaults to `''`

        :return: The RandomForestRegressor or RandomForestClassifier model fit on inputs and outputs.
        :rtype: RandomForest
        """
        criterion = cls._check_criterion(criterion)
        outputs = matrix_1d_to_vector(outputs)
        return cls.fit_(inputs, outputs, name=name, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                        n_jobs=1)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, n_estimators: int = 100,
                    criterion: str = 'squared_error', max_depth: int = None,
                    name: str = '') -> Tuple['RandomForest', Union[Vector, Matrix]]:
        """
        Fit outputs to inputs and predict outputs for predicted_inputs. It is the succession of fit and predict methods.
        """
        criterion = cls._check_criterion(criterion)
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, n_estimators=n_estimators,
                                criterion=criterion, max_depth=max_depth, n_jobs=1)


class RandomForestRegressor(RandomForest):
    """
    Base class for handling scikit-learn RandomForestRegressor.

    Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for
    more information on RandomForestRegressor.

    :param n_outputs_:
        The number of outputs when fit is performed.
    :type n_outputs_: int, `optional`, defaults to `None`

    :param estimators_:
        The collection of fitted sub-trees.
    :type estimators_: List[DecisionTreeRegressor], `optional`, defaults to `None`

    :param name:
        Name of RandomForestRegressor
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, n_outputs_: int = None, estimators_: List[DecisionTreeRegressor] = None, name: str = ''):
        RandomForest.__init__(self, estimators_=estimators_, n_outputs_=n_outputs_, name=name)

    @classmethod
    def _skl_class(cls):
        return ensemble.RandomForestRegressor

    def _instantiate_skl(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        kwargs = cls.generic_dessia_attributes(model, name=name)
        kwargs['estimators_'] = [DecisionTreeRegressor._instantiate_dessia(tree) for tree in model.estimators_]
        return cls(**kwargs)


class RandomForestClassifier(RandomForest):
    """
    Base class for handling scikit-learn RandomForestClassifier.

    Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for
    more information on RandomForestClassifier.

    :param n_classes_:
        The number of classes (for single output problems), or a list containing the number of classes for each output
        (for multi-output problems).
    :type n_classes_: Union[int, List[int]], `optional`, defaults to `None`

    :param classes_:
        The number of outputs when fit is performed.
    :type classes_: List[int], `optional`, defaults to `None`

    :param n_outputs_:
        The number of outputs when fit is performed.
    :type n_outputs_: int, `optional`, defaults to `None`

    :param estimators_:
        The collection of fitted sub-trees.
    :type estimators_: List[DecisionTreeClassifier], `optional`, defaults to `None`

    :param name:
        Name of RandomForestClassifier
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, n_classes_: int = None, classes_: List[int] = None, n_outputs_: int = None,
                 estimators_: List[DecisionTreeRegressor] = None, name: str = ''):
        self.n_classes_ = n_classes_
        self.classes_ = classes_
        RandomForest.__init__(self, estimators_=estimators_, n_outputs_=n_outputs_, name=name)

    @classmethod
    def _skl_class(cls):
        return ensemble.RandomForestClassifier

    def _instantiate_skl(self):
        model = self.generic_skl_attributes()
        model.n_classes_ = self.n_classes_
        model.classes_ = npy.array(self.classes_)
        return model

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        kwargs = cls.generic_dessia_attributes(model, name=name)
        kwargs.update({'estimators_': [DecisionTreeClassifier._instantiate_dessia(tree) for tree in model.estimators_],
                       'n_classes_': int(model.n_classes_),
                       'classes_': model.classes_.tolist()})
        return cls(**kwargs)


class SupportVectorMachine(Model):
    """
    Base object for handling a scikit-learn SupportVectorMachine objects.

    Please refer to https://scikit-learn.org/stable/modules/svm.html for more
    information on SupportVectorMachine object and understanding the SupportVectorMachine for basic usage.

    :param kernel:
        Specifies the kernel type to be used in the algorithm.
        Can be one of `[‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]`. If `None` is given, ‘rbf’ will be used.
        If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be
        an matrix of shape `n_samples x n_samples`
    :type kernel: str, `optional`, defaults to `'rbf'`

    :param raw_coef_:
        The number of classes (for single output problems), or a list containing the number of classes for each output
        (for multi-output problems).
    :type raw_coef_: List[List[float]], `optional`, defaults to `None`

    :param _dual_coef_:
        Coefficients of the support vector in the decision function. Shape is `1 x _n_support`.
    :type _dual_coef_: List[List[float]], `optional`, defaults to `None`

    :param _intercept_:
        Constants in decision function.
    :type _intercept_: List[float], `optional`, defaults to `None`

    :param support_:
        Indices of support vectors.
    :type support_: List[int], `optional`, defaults to `None`

    :param support_vectors_:
        Support vectors.
    :type support_vectors_: List[List[float]], `optional`, defaults to `None`

    :param _n_support:
        Number of support vectors for each class.
    :type _n_support: List[int], `optional`, defaults to `None`

    :param _probA:
        Parameter learned in Platt scaling when `probability=True`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.probA_
    :type _probA: List[float], `optional`, defaults to `None`

    :param _probB:
        Parameter learned in Platt scaling when `probability=True`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.probB_
    :type _probB: List[float], `optional`, defaults to `None`

    :param _gamma:
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
           - if `gamma='scale'` (default) is passed then it uses `1 / (n_features * X.var())` as value of gamma,
           - if `‘auto’`, uses `1 / n_features`.
    :type _gamma: float, `optional`, defaults to `1.`

    :param _sparse:
        Specify if the inputs are a sparse matrix or not.
    :type _sparse: bool, `optional`, defaults to `False`
    """

    def __init__(self, kernel: str = 'rbf', raw_coef_: Matrix = None, _dual_coef_: Matrix = None,
                 _intercept_: Vector = None, support_: List[int] = 1, support_vectors_: Matrix = None,
                 _n_support: List[int] = None, _probA: Vector = None, _probB: Vector = None,
                 _gamma: float = 1., _sparse: bool = False, name: str = ''):
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
        Model.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for SupportVectorMachine. Please use '\
                                  'SupportVectorClassifier or SupportVectorRegressor.')

    def _call_skl_model(self):
        return self._skl_class()(kernel=self.kernel)

    def generic_skl_attributes(self):
        """
        Generic method (shared between SupportVectorMachine) to set scikit-learn model attributes from self attributes.
        """
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
        """
        Generic method (shared between SupportVectorMachine) to set self attributes from scikit-learn model attributes.
        """
        return {'name': name,
                'kernel': model.kernel,
                'raw_coef_': model._get_coef().tolist(),
                '_dual_coef_': model._dual_coef_.tolist(),
                '_intercept_': model._intercept_.tolist(),
                'support_': model.support_.tolist(),
                'support_vectors_': model.support_vectors_.tolist(),
                '_n_support': model._n_support.tolist(),
                '_probA': model._probA.tolist(),
                '_probB': model._probB.tolist(),
                '_gamma': float(model._gamma),
                '_sparse': model._sparse}

    @classmethod
    def init_for_modeler(cls, C: float = 1., kernel: str = 'rbf') -> Tuple['SupportVectorMachine', Dict[str, Any], str]:
        """
        Initialize class SupportVectorMachine with its name and hyperparemeters to fit in Modeler.

        :param C:
            Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty.
        :type C: float, `optional`, defaults to `1.`

        :param kernel:
            Specifies the kernel type to be used in the algorithm.
            Can be one of `[‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]`. If `None` is given, ‘rbf’ will be used.
            If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be
            an matrix of shape `n_samples x n_samples`
        :type kernel: str, `optional`, defaults to `'rbf'`

        :return: The SupportVectorMachine model fit on inputs and outputs.
        :rtype: Tuple['SupportVectorMachine', Dict[str, Any], str]
        """
        return cls.init_for_modeler_(C=C, kernel=kernel)

    @classmethod
    def fit(cls, inputs: Matrix, outputs: Vector, C: float = 1., kernel: str = 'rbf',
            name: str = '') -> 'SupportVectorMachine':
        """
        Standard method to fit outputs to inputs thanks to SupportVectorMachine model from scikit-learn.

        More information:
            - Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            - Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param C:
            Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty.
        :type C: float, `optional`, defaults to `1.`

        :param kernel:
            Specifies the kernel type to be used in the algorithm.
            Can be one of `[‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]`. If `None` is given, ‘rbf’ will be used.
            If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be
            an matrix of shape `n_samples x n_samples`
        :type kernel: str, `optional`, defaults to `'rbf'`

        :param name:
            Name of SupportVectorRegressor or SupportVectorClassifier model
        :type name: str, `optional`, defaults to `''`

        :return: The SupportVectorRegressor or SupportVectorClassifier model fit on inputs and outputs.
        :rtype: SupportVectorMachine
        """
        return cls.fit_(inputs, outputs, name=name, C=C, kernel=kernel)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Vector, predicted_inputs: Matrix, C: float = 1., kernel: str = 'rbf',
                    name: str = '') -> Tuple['SupportVectorMachine', Union[Vector, Matrix]]:
        """
        Fit outputs to inputs and predict outputs for predicted_inputs. It is the succession of fit and predict methods.
        """
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, C=C, kernel=kernel)


class SupportVectorRegressor(SupportVectorMachine):
    """
    Base object for handling a scikit-learn SupportVectorRegressor objects.

    Please refer to https://scikit-learn.org/stable/modules/svm.html#svm-regression for more
    information on SupportVectorRegressor object and understanding the SupportVectorRegressor for basic usage.

    :param kernel:
        Specifies the kernel type to be used in the algorithm.
        Can be one of `[‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]`. If `None` is given, ‘rbf’ will be used.
        If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be
        an matrix of shape `n_samples x n_samples`
    :type kernel: str, `optional`, defaults to `'rbf'`

    :param raw_coef_:
        The number of classes (for single output problems), or a list containing the number of classes for each output
        (for multi-output problems).
    :type raw_coef_: List[List[float]], `optional`, defaults to `None`

    :param _dual_coef_:
        Coefficients of the support vector in the decision function. Shape is `1 x _n_support`.
    :type _dual_coef_: List[List[float]], `optional`, defaults to `None`

    :param _intercept_:
        Constants in decision function.
    :type _intercept_: List[float], `optional`, defaults to `None`

    :param support_:
        Indices of support vectors.
    :type support_: List[int], `optional`, defaults to `None`

    :param support_vectors_:
        Support vectors.
    :type support_vectors_: List[List[float]], `optional`, defaults to `None`

    :param _n_support:
        Number of support vectors for each class.
    :type _n_support: List[int], `optional`, defaults to `None`

    :param _probA:
        Parameter learned in Platt scaling when `probability=True`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.probA_
    :type _probA: List[float], `optional`, defaults to `None`

    :param _probB:
        Parameter learned in Platt scaling when `probability=True`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.probB_
    :type _probB: List[float], `optional`, defaults to `None`

    :param _gamma:
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
           - if `gamma='scale'` (default) is passed then it uses `1 / (n_features * X.var())` as value of gamma,
           - if `‘auto’`, uses `1 / n_features`.
    :type _gamma: float, `optional`, defaults to `1.`

    :param _sparse:
        Specify if the inputs are a sparse matrix or not.
    :type _sparse: bool, `optional`, defaults to `False`
    """
    _standalone_in_db = True

    def __init__(self, kernel: str = 'rbf', raw_coef_: Matrix = None, _dual_coef_: Matrix = None,
                 _intercept_: Vector = None, support_: List[int] = 1, support_vectors_: Matrix = None,
                 _n_support: List[int] = None, _probA: Vector = None, _probB: Vector = None, _gamma: float = 1.,
                 _sparse: bool = False, name: str = ''):
        SupportVectorMachine.__init__(self, raw_coef_=raw_coef_, _dual_coef_=_dual_coef_,
                                      support_vectors_=support_vectors_, _sparse=_sparse, kernel=kernel,
                                      _n_support=_n_support, support_=support_, _intercept_=_intercept_, _probA=_probA,
                                      _probB=_probB, _gamma=_gamma, name=name)

    @classmethod
    def _skl_class(cls):
        return svm.SVR

    def _instantiate_skl(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        return cls(**cls.generic_dessia_attributes(model, name=name))


class SupportVectorClassifier(SupportVectorMachine):
    """
    Base object for handling a scikit-learn SupportVectorClassifier objects.

    Please refer to https://scikit-learn.org/stable/modules/svm.html#svm-classification for more
    information on SupportVectorClassifier object and understanding the SupportVectorClassifier for basic usage.

    :param kernel:
        Specifies the kernel type to be used in the algorithm.
        Can be one of `[‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’]`. If `None` is given, ‘rbf’ will be used.
        If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be
        an matrix of shape `n_samples x n_samples`
    :type kernel: str, `optional`, defaults to `'rbf'`

    :param raw_coef_:
        The number of classes (for single output problems), or a list containing the number of classes for each output
        (for multi-output problems).
    :type raw_coef_: List[List[float]], `optional`, defaults to `None`

    :param _dual_coef_:
        Coefficients of the support vector in the decision function. Shape is `1 x _n_support`.
    :type _dual_coef_: List[List[float]], `optional`, defaults to `None`

    :param _intercept_:
        Constants in decision function.
    :type _intercept_: List[float], `optional`, defaults to `None`

    :param support_:
        Indices of support vectors.
    :type support_: List[int], `optional`, defaults to `None`

    :param support_vectors_:
        Support vectors.
    :type support_vectors_: List[List[float]], `optional`, defaults to `None`

    :param _n_support:
        Number of support vectors for each class.
    :type _n_support: List[int], `optional`, defaults to `None`

    :param _probA:
        Parameter learned in Platt scaling when `probability=True`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.probA_
    :type _probA: List[float], `optional`, defaults to `None`

    :param _probB:
        Parameter learned in Platt scaling when `probability=True`.
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.probB_
    :type _probB: List[float], `optional`, defaults to `None`

    :param _gamma:
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
           - if `gamma='scale'` (default) is passed then it uses `1 / (n_features * X.var())` as value of gamma,
           - if `‘auto’`, uses `1 / n_features`.
    :type _gamma: float, `optional`, defaults to `1.`

    :param _sparse:
        Specify if the inputs are a sparse matrix or not.
    :type _sparse: bool, `optional`, defaults to `False`

    :param classes_:
        The classes labels.
    :type classes_: List[int], `optional`, defaults to `None`
    """

    def __init__(self, kernel: str = 'rbf', raw_coef_: Matrix = None, _dual_coef_: Matrix = None,
                 _intercept_: Vector = None, support_: List[int] = 1, support_vectors_: Matrix = None,
                 _n_support: List[int] = None, _probA: Vector = None, _probB: Vector = None,
                 _gamma: float = 1., _sparse: bool = False, classes_: List[int] = None, name: str = ''):
        self.classes_ = classes_
        SupportVectorMachine.__init__(self, raw_coef_=raw_coef_, _dual_coef_=_dual_coef_,
                                      support_vectors_=support_vectors_, _sparse=_sparse, kernel=kernel,
                                      _n_support=_n_support, support_=support_, _intercept_=_intercept_, _probA=_probA,
                                      _probB=_probB, _gamma=_gamma, name=name)

    @classmethod
    def _skl_class(cls):
        return svm.SVC

    def _instantiate_skl(self):
        model = self.generic_skl_attributes()
        model.classes_ = npy.array(self.classes_)
        return model

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        kwargs = cls.generic_dessia_attributes(model, name=name)
        kwargs['classes_'] = model.classes_.tolist()
        return cls(**kwargs)


class MultiLayerPerceptron(Model):
    """
    Base object for handling a scikit-learn MultiLayerPerceptron (dense neural network).

    Please refer to https://scikit-learn.org/stable/modules/neural_networks_supervised.html for more
    information on MultiLayerPerceptron.

    :param coef_:
        List of coefficients of the model.
    :type coef_: List[List[float]], `optional`, defaults to `None`

    :param intercept_:
        List of offsets of the model.
    :type intercept_: List[List[float]], `optional`, defaults to `None`

    :param n_layers_:
        Number of hidden layers contained in the current MultiLayerPerceptron.
    :type n_layers_: int, `optional`, defaults to `None`

    :param activation:
        Activation function for hidden layers.
    :type activation: str, `optional`, defaults to `'relu'`

    :param out_activation_:
        Activation function for the output layer.
    :type out_activation_: str, `optional`, defaults to `'identity'`

    :param name:
        Name of MultiLayerPerceptron
    :type name: str, `optional`, defaults to `''`
    """

    def __init__(self, coefs_: List[Matrix] = None, intercepts_: Matrix = None,
                 n_layers_: int = None, activation: str = 'relu', out_activation_: str = 'identity', name: str = ''):
        self.coefs_ = coefs_
        self.intercepts_ = intercepts_
        self.n_layers_ = n_layers_
        self.activation = activation
        self.out_activation_ = out_activation_
        Model.__init__(self, name=name)

    @classmethod
    def _skl_class(cls):
        raise NotImplementedError('Method _skl_class not implemented for MultiLayerPerceptron. Please use '\
                                  'MLPRegressor or MLPClassifier.')

    def generic_skl_attributes(self):
        """
        Generic method (shared between MultiLayerPerceptron) to set scikit-learn model attributes from self attributes.
        """
        model = self._call_skl_model()
        model.coefs_ = [npy.array(coefs_) for coefs_ in self.coefs_]
        model.intercepts_ = [npy.array(intercepts_) for intercepts_ in self.intercepts_]
        model.n_layers_ = self.n_layers_
        model.activation = self.activation
        model.out_activation_ = self.out_activation_
        return model

    @classmethod
    def generic_dessia_attributes(cls, model, name: str = ''):
        """
        Generic method (shared between MultiLayerPerceptron) to set self attributes from scikit-learn model attributes.
        """
        return {'name': name,
                'coefs_': [coefs_.tolist() for coefs_ in model.coefs_],
                'intercepts_': [intercepts_.tolist() for intercepts_ in model.intercepts_],
                'n_layers_': model.n_layers_,
                'activation': model.activation,
                'out_activation_': model.out_activation_}

    @classmethod
    def init_for_modeler(cls, hidden_layer_sizes: List[int] = None, activation: str = 'relu', alpha: float = 0.0001,
                         solver: str = 'adam', max_iter: int = 200,
                         tol: float = 0.0001) -> Tuple['MultiLayerPerceptron', Dict[str, Any], str]:
        """
        Initialize class MultiLayerPerceptron with its name and hyperparemeters to fit in Modeler.

        :param hidden_layer_sizes:
            Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty.
        :type hidden_layer_sizes: List[int], `optional`, defaults to `None`

        :param activation:
            Activation function for the hidden layer:
                - `‘identity’`, no-op activation, useful to implement linear bottleneck, returns `f(x) = x`
                - `‘logistic’`, the logistic sigmoid function, returns `f(x) = 1 / (1 + exp(-x))`.
                - `‘tanh’`, the hyperbolic tan function, returns `f(x) = tanh(x)`.
                - `‘relu’`, the rectified linear unit function, returns `f(x) = max(0, x)`
        :type activation: str, `optional`, defaults to `'relu'`

        :param alpha:
            Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative
            float i.e. in [0, inf[.
        :type alpha: float, `optional`, defaults to `0.0001`

        :param solver:
            The solver for weight optimization:
                - `‘lbfgs’` is an optimizer in the family of quasi-Newton methods.
                - `‘sgd’` refers to stochastic gradient descent.
                - `‘adam’` refers to a stochastic gradient-based optimizer proposed in https://arxiv.org/abs/1412.6980
            Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training
            samples or more) in terms of both training time and validation score. For small datasets, however,
            `‘lbfgs’` can converge faster and perform better.
        :type solver: str, `optional`, defaults to `'adam'`

        :param max_iter:
            Maximum number of iterations. The solver iterates until convergence (determined by `‘tol’`) or this number
            of iterations. For stochastic solvers (`‘sgd’`, `‘adam’`), note that this determines the number of epochs
            (how many times each data point will be used), not the number of gradient steps.
        :type max_iter: int, `optional`, defaults to `200`

        :param tol:
            Tolerance for the optimization. When the loss or score is not improving by at least tol for
            `n_iter_no_change` consecutive iterations, unless `learning_rate` is set to `‘adaptive’`, convergence is
            considered to be reached and training stops.
        :type tol: float, `optional`, defaults to `0.0001`

        :return: The MultiLayerPerceptron model fit on inputs and outputs.
        :rtype: Tuple['MultiLayerPerceptron', Dict[str, Any], str]
        """
        return cls.init_for_modeler_(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha,
                                     solver=solver, max_iter=max_iter, tol=tol)

    @classmethod
    def fit(cls, inputs: Matrix, outputs: Vector, hidden_layer_sizes: List[int] = None,
            activation: str = 'relu', alpha: float = 0.0001, solver: str = 'adam', max_iter: int = 200,
            tol: float = 0.0001, name: str = '') -> 'MultiLayerPerceptron':
        """
        Standard method to fit outputs to inputs thanks to MLPRegressor or MLPClassifier models from scikit-learn.

        More information:
            - Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
            - Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param hidden_layer_sizes:
            Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty.
        :type hidden_layer_sizes: List[int], `optional`, defaults to `None`

        :param activation:
            Activation function for the hidden layer:
                - `‘identity’`, no-op activation, useful to implement linear bottleneck, returns `f(x) = x`
                - `‘logistic’`, the logistic sigmoid function, returns `f(x) = 1 / (1 + exp(-x))`.
                - `‘tanh’`, the hyperbolic tan function, returns `f(x) = tanh(x)`.
                - `‘relu’`, the rectified linear unit function, returns `f(x) = max(0, x)`
        :type activation: str, `optional`, defaults to `'relu'`

        :param alpha:
            Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative
            float i.e. in [0, inf[.
        :type alpha: float, `optional`, defaults to `0.0001`

        :param solver:
            The solver for weight optimization:
                - `‘lbfgs’` is an optimizer in the family of quasi-Newton methods.
                - `‘sgd’` refers to stochastic gradient descent.
                - `‘adam’` refers to a stochastic gradient-based optimizer proposed in https://arxiv.org/abs/1412.6980
            Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training
            samples or more) in terms of both training time and validation score. For small datasets, however,
            `‘lbfgs’` can converge faster and perform better.
        :type solver: str, `optional`, defaults to `'adam'`

        :param max_iter:
            Maximum number of iterations. The solver iterates until convergence (determined by `‘tol’`) or this number
            of iterations. For stochastic solvers (`‘sgd’`, `‘adam’`), note that this determines the number of epochs
            (how many times each data point will be used), not the number of gradient steps.
        :type max_iter: int, `optional`, defaults to `200`

        :param tol:
            Tolerance for the optimization. When the loss or score is not improving by at least tol for
            `n_iter_no_change` consecutive iterations, unless `learning_rate` is set to `‘adaptive’`, convergence is
            considered to be reached and training stops.
        :type tol: float, `optional`, defaults to `0.0001`

        :return: The MLPRegressor or MLPClassifier model fit on inputs and outputs.
        :rtype: MultiLayerPerceptron
        """
        outputs = matrix_1d_to_vector(outputs)
        return cls.fit_(inputs, outputs, name=name, hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                        alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Vector, predicted_inputs: Matrix,
                    hidden_layer_sizes: List[int] = None, activation: str = 'relu', alpha: float = 0.0001,
                    solver: str = 'adam', max_iter: int = 200, tol: float = 0.0001,
                    name: str = '') -> Tuple['MultiLayerPerceptron', Union[Vector, Matrix]]:
        """
        Fit outputs to inputs and predict outputs for predicted_inputs. It is the succession of fit and predict methods.
        """
        return cls.fit_predict_(inputs, outputs, predicted_inputs, name=name, hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation, alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)


class MLPRegressor(MultiLayerPerceptron):
    """
    Base object for handling a scikit-learn MLPRegressor (dense neural network) object.

    Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html for more
    information on MLPRegressor.

    :param coef_:
        List of coefficients of the model.
    :type coef_: List[List[float]], `optional`, defaults to `None`

    :param intercept_:
        List of offsets of the model.
    :type intercept_: List[List[float]], `optional`, defaults to `None`

    :param n_layers_:
        Number of hidden layers contained in the current MLPRegressor.
    :type n_layers_: int, `optional`, defaults to `None`

    :param activation:
        Activation function for hidden layers.
    :type activation: str, `optional`, defaults to `'relu'`

    :param out_activation_:
        Activation function for the output layer.
    :type out_activation_: str, `optional`, defaults to `'identity'`

    :param name:
        Name of MLPRegressor
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, coefs_: List[Matrix] = None, intercepts_: Matrix = None,
                 n_layers_: int = None, activation: str = 'relu', out_activation_: str = 'identity', name: str = ''):
        MultiLayerPerceptron.__init__(self, coefs_=coefs_, intercepts_=intercepts_, n_layers_=n_layers_,
                                      activation=activation, out_activation_=out_activation_, name=name)

    @classmethod
    def _skl_class(cls):
        return neural_network.MLPRegressor

    def _instantiate_skl(self):
        return self.generic_skl_attributes()

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        return cls(**cls.generic_dessia_attributes(model, name=name))


class MLPClassifier(MultiLayerPerceptron):
    """
    Base object for handling a scikit-learn MLPClassifier (dense neural network) object.

    Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html for more
    information on MLPClassifier.

    :param coef_:
        List of coefficients of the model.
    :type coef_: List[List[float]], `optional`, defaults to `None`

    :param intercept_:
        List of offsets of the model.
    :type intercept_: List[List[float]], `optional`, defaults to `None`

    :param n_layers_:
        Number of hidden layers contained in the current MLPClassifier.
    :type n_layers_: int, `optional`, defaults to `None`

    :param activation:
        Activation function for hidden layers.
    :type activation: str, `optional`, defaults to `'relu'`

    :param out_activation_:
        Activation function for the output layer.
    :type out_activation_: str, `optional`, defaults to `'identity'`

    :param n_outputs_:
        The number of outputs when fit is performed.
    :type n_outputs_: int, `optional`, defaults to `None`

    :param _label_binarizer:
        Data scaler used in MLPClassifier to standardize class labels.
    :type _label_binarizer: LabelBinarizer, `optional`, defaults to `None`

    :param name:
        Name of MLPClassifier
    :type name: str, `optional`, defaults to `''`
    """
    _standalone_in_db = True

    def __init__(self, coefs_: List[Matrix] = None, intercepts_: Matrix = None,
                 n_layers_: int = None, activation: str = 'relu', out_activation_: str = 'identity',
                 n_outputs_: int = None, _label_binarizer: LabelBinarizer = None, name: str = ''):
        self.n_outputs_ = n_outputs_
        self._label_binarizer = _label_binarizer
        MultiLayerPerceptron.__init__(self, coefs_=coefs_, intercepts_=intercepts_, n_layers_=n_layers_,
                                      activation=activation, out_activation_=out_activation_, name=name)

    @classmethod
    def _skl_class(cls):
        return neural_network.MLPClassifier

    def _instantiate_skl(self):
        model = self.generic_skl_attributes()
        model.n_outputs_ = self.n_outputs_
        model._label_binarizer = self._label_binarizer.instantiate_skl()
        return model

    @classmethod
    def _instantiate_dessia(cls, model, name: str = ''):
        kwargs = cls.generic_dessia_attributes(model, name=name)
        kwargs.update({'n_outputs_': model.n_outputs_,
                       '_label_binarizer': LabelBinarizer.instantiate_dessia(model._label_binarizer)})
        return cls(**kwargs)


def get_scaler_attr(scaler, attr: str):
    """
    Get attribute attr of scikit-learn scaler with an exception for numpy arrays (in order to instantiate a Scaler).
    """
    scaler_attr = getattr(scaler, attr)
    if isinstance(scaler_attr, npy.ndarray):
        return scaler_attr.tolist()
    return scaler_attr

def get_split_indexes(len_matrix: int, ratio: float = 0.8, shuffled: bool = True) -> Tuple[Vector, Vector]:
    """
    Get two lists of indexes to split randomly a matrix in two matrices.
    """
    if ratio > 1:
        len_train = int(ratio)
    else:
        len_train = int(len_matrix * ratio)

    idx_range = range(0, len_matrix)
    ind_train = random.sample(idx_range, len_train)
    ind_test = list(set(idx_range).difference(set(ind_train)))

    if not shuffled:
        ind_train.sort()
        ind_test.sort()
    return ind_train, ind_test

def train_test_split(*matrices: List[Matrix], ratio: float = 0.8, shuffled: bool = True) -> List[Matrix]:
    """
    Split a list of matrices of the same length into a list of train and test split matrices.

    The first one is of length `int(len_matrix * ratio)`, the second of length `len_matrix - int(len_matrix * ratio)`.

    :param len_matrix:
        Length of matrix to split.
    :type len_matrix: List[List[float]]

    :param ratio:
        Ratio on which to split matrix. If ratio > 1, ind_train will be of length `int(ratio)` and ind_test of
        length `len_matrix - int(ratio)`.
    :type ratio: float, `optional`, defaults to 0.8

    :param shuffled:
        Whether to shuffle or not the results.
    :type shuffled: bool, `optional`, defaults to True

    :return:
        A list containing all split matrices in the following order: `[train_M1, test_M1, train_M2, test_M2, ...,
        train_Mn, test_Mn]`.
    :rtype: List[List[List[float]]]
    """
    len_matrices = [len(matrix) for matrix in matrices]
    if len(set(len_matrices)) != 1:
        raise ValueError("matrices are not of the same length in train_test_split.")

    ind_train, ind_test = get_split_indexes(len_matrices[0], ratio=ratio, shuffled=shuffled)
    train_test_split_matrices = [[[matrix[idx] for idx in ind_train], [matrix[idx] for idx in ind_test]]
                                 for matrix in matrices]
    return sum(train_test_split_matrices, [])

def matrix_1d_to_vector(matrix: Matrix) -> Union[Vector, Matrix]:
    """
    Transform a List[List[float]] of shape (n, 1) into a List[float] of shape (n,).
    """
    if isinstance(matrix[0], list):
        if len(matrix[0]) == 1:
            return sum(matrix, [])
    return matrix

def vector_to_2d_matrix(matrix: Union[Vector, Matrix]) -> Matrix:
    """
    Transform a List[float] of shape (n,) into a List[List[float]] of shape (n, 1).
    """
    if not isinstance(matrix[0], list):
        return [[x] for x in matrix]
    return matrix


