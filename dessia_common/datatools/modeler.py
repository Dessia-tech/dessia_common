"""
Librairy for building machine learning modelers from Dataset or Lists using sklearn models handled in models.
"""
from typing import List, Dict, Any, Tuple, Type, Union

import numpy as npy

from plot_data.core import Dataset as pl_Dataset
from plot_data.core import EdgeStyle, Tooltip, MultiplePlots, PointStyle, Graph2D, Axis
from plot_data.colors import BLACK, RED, BLUE, WHITE

from dessia_common.core import DessiaObject
from dessia_common.datatools import models
from dessia_common.datatools.dataset import Dataset

Vector = List[float]
Matrix = List[Vector]
Points = List[Dict[str, float]]

NO_LINE = EdgeStyle(0.0001)
STD_LINE = EdgeStyle(line_width=1.5, color_stroke=BLACK)

REF_POINT_STYLE = PointStyle(BLUE, BLUE, 0.1, 2., 'circle')
VAL_POINT_STYLE = PointStyle(RED, RED, 0.1, 2., 'circle')
LIN_POINT_STYLE = PointStyle(BLACK, BLACK, 0.1, 1, 'crux')
INV_POINT_STYLE = PointStyle(WHITE, WHITE, 0.1, 1, 'crux')


class Modeler(DessiaObject):
    """
    Object that encapsulate standard processes in machine learning modelisations.

    Modeler object allows to:
        * fit a model from models
        * prescale input and output data before fit or predict
        * score a model from models
        * validate a modelisation process with cross_validation method
        * plot performances and predictions of a model stored in Modeler
        * store a fitted model and associated fitted scalers in a Modeler element that can be re-used in another
        workflow as an already trained machine learning model

    :param model:
        Fitted model to make predictions.
    :type model: models.Model

    :param input_scaler:
        Scaler for input data.
    :type input_scaler: models.Scaler

    :param output_scaler:
        caler for output data.
    :type output_scaler: models.Scaler

    :param name:
        Name of Modeler.
    :type name: str, `optional`, defaults to `''`
    """

    def __init__(self, model: models.Model, input_scaler: models.Scaler, output_scaler: models.Scaler, name: str = ''):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.in_scaled = self._is_scaled(self.input_scaler)
        self.out_scaled = self._is_scaled(self.output_scaler)
        DessiaObject.__init__(self, name=name)

    def _is_scaled(self, scaler: models.Scaler):
        if isinstance(scaler, models.IdentityScaler):
            return False
        return True

    def _format_output(self, scaled_outputs: Matrix):
        """
        Format output to List[List[float]] in any case for code consistency and simplicity.
        """
        if not isinstance(scaled_outputs[0], list):
            return [[value] for value in scaled_outputs]
        return scaled_outputs

    @staticmethod
    def _compute_scalers(inputs: Matrix, outputs: Matrix, input_is_scaled: bool = True, output_is_scaled: bool = False,
                         name: str = '') -> Tuple[models.Scaler, models.Scaler, Matrix, Matrix]:
        in_scaler_class, input_scaler_name = models.Scaler.set_in_modeler(name, "in", input_is_scaled)
        out_scaler_class, output_scaler_name = models.Scaler.set_in_modeler(name, "out", output_is_scaled)

        in_scaler, scaled_inputs = in_scaler_class.fit_transform(inputs, input_scaler_name)
        out_scaler, scaled_outputs = out_scaler_class.fit_transform(outputs, output_scaler_name)
        return in_scaler, out_scaler, scaled_inputs, scaled_outputs

    @classmethod
    def _fit(cls, inputs: Matrix, outputs: Matrix, model: models.Model, input_is_scaled: bool = True,
             output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Private method to fit outputs to inputs with a machine learning method from datatools.models objects.
        """
        in_scaler, out_scaler, scaled_inputs, scaled_outputs = cls._compute_scalers(inputs, outputs, input_is_scaled,
                                                                                    output_is_scaled, name)
        fit_model = model.fit(scaled_inputs, scaled_outputs, **model.params, name=name + '_model')
        return cls(fit_model, in_scaler, out_scaler, name)

    @classmethod
    def fit_matrix(cls, inputs: Matrix, outputs: Matrix, model: models.Model, input_is_scaled: bool = True,
                   output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Method to fit outputs to inputs with a machine learning method from datatools.models objects for matrix data.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param input_is_scaled:
            Whether to standardize inputs or not with a models.StandardScaler
        :type input_is_scaled: bool, `optional`, True

        :param output_is_scaled:
            Whether to standardize outputs or not with a models.StandardScaler
        :type output_is_scaled: bool, `optional`, False

        :param name:
            Name of Modeler
        :type name: str, `optional`, defaults to `''`

        :return: The equivalent Modeler object containing the fitted model and scalers associated to inputs and outputs
        :rtype: Modeler
        """
        return cls._fit(inputs, outputs, model, input_is_scaled, output_is_scaled, name)

    @classmethod
    def fit_dataset(cls, dataset: Dataset, input_names: List[str], output_names: List[str], model: models.Model,
                    input_is_scaled: bool = True, output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Method to fit outputs to inputs with a machine learning method from datatools.models objects for a Dataset.

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features
        :type input_names: List[str]

        :param output_names:
            Names of output features
        :type output_names: List[str]

        :param input_is_scaled:
            Whether to standardize inputs or not with a models.StandardScaler
        :type input_is_scaled: bool, `optional`, True

        :param output_is_scaled:
            Whether to standardize outputs or not with a models.StandardScaler
        :type output_is_scaled: bool, `optional`, False

        :param name:
            Name of Modeler
        :type name: str, `optional`, defaults to `''`

        :return: The equivalent Modeler object containing the fitted model and scalers associated to inputs and outputs
        :rtype: Modeler
        """
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return cls.fit_matrix(inputs, outputs, model, input_is_scaled, output_is_scaled, name)

    def _predict(self, inputs: List[List[float]]) -> Union[Vector, Matrix]:
        """
        Private method to predict outputs from inputs with self.model.
        """
        return self.output_scaler.inverse_transform(self.model.predict(self.input_scaler.transform(inputs)))

    def predict_matrix(self, inputs: List[List[float]]) -> Matrix:
        """
        Method to predict outputs from inputs with the current Modeler for matrix data.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :return: The predicted values for inputs.
        :rtype: List[List[float]]
        """
        return self._format_output(self._predict(inputs))

    def predict_dataset(self, dataset: Dataset, input_names: List[str]) -> Matrix:
        """
        Method to predict outputs from inputs with the current Modeler for Dataset object.

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features to predict
        :type input_names: List[str]

        :return: The predicted values for inputs.
        :rtype: List[List[float]]
        """
        inputs = dataset.sub_matrix(input_names)
        return self.predict_matrix(inputs)

    @classmethod
    def _fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_outputs: Matrix, model: models.Model,
                     input_is_scaled: bool = True, output_is_scaled: bool = False,
                     name: str = '') -> Tuple['Modeler', Union[Vector, Matrix]]:
        """
        Private method to fit outputs to inputs and predict predicted_outputs for a Dataset (fit then predict).
        """
        modeler = cls._fit(inputs, outputs, model, input_is_scaled, output_is_scaled, name)
        return modeler, modeler._predict(predicted_outputs)

    @classmethod
    def fit_predict_matrix(cls, inputs: Matrix, outputs: Matrix, predicted_outputs: Matrix, model: models.Model,
                           input_is_scaled: bool = True, output_is_scaled: bool = False,
                           name: str = '') -> Tuple['Modeler', Matrix]:
        """
        Fit outputs to inputs and predict predicted_outputs for matrix data (fit then predict).
        """
        modeler, predictions = cls._fit_predict(inputs, outputs, predicted_outputs, model, input_is_scaled,
                                                output_is_scaled, name)
        return modeler, modeler._format_output(predictions)

    @classmethod
    def fit_predict_dataset(cls, fit_dataset: Dataset, to_predict_dataset: Dataset, input_names: List[str],
                            output_names: List[str], model: models.Model, input_is_scaled: bool = True,
                            output_is_scaled: bool = False, name: str = '') -> Tuple['Modeler', Matrix]:
        """
        Fit outputs to inputs and predict outputs of to_predict_dataset (fit then predict).
        """
        modeler = cls.fit_dataset(fit_dataset, input_names, output_names, model, input_is_scaled, output_is_scaled,
                                  name)
        return modeler, modeler.predict_dataset(to_predict_dataset, input_names)

    def _score(self, inputs: Matrix, outputs: Matrix) -> float:
        """
        Compute the score of Modeler.
        """
        return self.model.score(self.input_scaler.transform(inputs), self.output_scaler.transform(outputs))

    def score_matrix(self, inputs: Matrix, outputs: Matrix) -> float:
        """
        Compute the score of Modeler from matrix.

        Please be sure to fit the model before computing its score and use test data and not train data.
        Train data is data used to train the model and shall not be used to evaluate its quality.
        Test data is data used to test the model and must not be used to train (fit) it.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :return: The score of Modeler.
        :rtype: float
        """
        return self._score(inputs, outputs)

    def score_dataset(self, dataset: Dataset, input_names: List[str], output_names: List[str]) -> float:
        """
        Compute the score of Modeler from Dataset.

        Please be sure to fit the model before computing its score and use test data and not train data.
        Train data is data used to train the model and shall not be used to evaluate its quality.
        Test data is data used to test the model and must not be used to train (fit) it.

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features
        :type input_names: List[str]

        :param output_names:
            Names of output features
        :type output_names: List[str]

        :return: The score of Modeler.
        :rtype: float
        """
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return self._score(inputs, outputs)

    @classmethod
    def _fit_score(cls, inputs_train: Matrix, inputs_test: Matrix, outputs_train: Matrix, outputs_test: Matrix,
                   model: models.Model, input_is_scaled: bool, output_is_scaled: bool,
                   name: str) -> Tuple['Modeler', float]:
        """
        Private method to fit modeler with train matrices and test it with test matrices.
        """
        mdlr = cls._fit(inputs_train, outputs_train, model, input_is_scaled, output_is_scaled, name)
        return mdlr, mdlr._score(inputs_test, outputs_test)

    @classmethod
    def from_matrix_fit_score(cls, inputs: Matrix, outputs: Matrix, model: models.Model, input_is_scaled: bool,
                              output_is_scaled: bool, ratio: float = 0.8, name: str = '') -> Tuple['Modeler', float]:
        """
        Fit modeler with train matrices and test it with test matrices.
        """
        in_train, in_test, out_train, out_test = models.train_test_split(inputs, outputs, ratio=ratio)
        return cls._fit_score(in_train, in_test, out_train, out_test, model, input_is_scaled,
                              output_is_scaled, name)

    @classmethod
    def from_dataset_fit_score(cls, dataset: Dataset, input_names: List[str], output_names: List[str],
                               model: models.Model, input_is_scaled: bool = True, output_is_scaled: bool = False,
                               ratio: float = 0.8, name: str = '') -> Tuple['Modeler', float]:
        """
        Train test split dataset, fit modeler with train matrices and score it with test matrices.
        """
        train_test_matrices = dataset.train_test_split(input_names, output_names, ratio)
        return cls._fit_score(*train_test_matrices, model, input_is_scaled, output_is_scaled, name)

    def features_importance(self):
        """
        Future features_importance method, maybe to put in dataset.
        """
        return

    def features_mrmr(self):
        """
        Future features_mrmr method, maybe to put in dataset.
        """
        return


class ValidationData(DessiaObject):
    """
    Object that stores data as 6 matrices: input_train, output_train, pred_train, input_test, output_test and pred_test.

    :param input_train:
        Matrix of input data used to train the modeler to validate.
    :type input_train: List[List[float]]

    :param input_test:
        Matrix of input data used to test the modeler to validate.
    :type input_test: List[List[float]]

    :param output_train:
        Matrix of reference output data corresponding to input_train.
    :type output_train: List[List[float]]

    :param output_test:
        Matrix of reference output data corresponding to input_test.
    :type output_test: List[List[float]]

    :param pred_train:
        Matrix of prediction of input_train made with the modeler to validate.
    :type pred_train: List[List[float]]

    :param pred_test:
        Matrix of prediction of input_test made with the modeler to validate.
    :type pred_test: List[List[float]]

    :param input_names:
        Names of input features
    :type input_names: List[str]

    :param output_names:
        Names of output features
    :type output_names: List[str]

    :param name:
        Name of Modeler.
    :type name: str, `optional`, defaults to `''`
    """

    def __init__(self, input_train: Matrix, input_test: Matrix, output_train: Matrix, output_test: Matrix,
                 pred_train: Matrix, pred_test: Matrix,input_names: List[str], output_names: List[str], name: str = ''):
        self.input_train = input_train
        self.input_test = input_test
        self.output_train = output_train
        self.output_test = output_test
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.input_names = input_names
        self.output_names = output_names
        DessiaObject.__init__(self, name=name)

    def _concatenate_outputs(self) -> Matrix:
        return self.output_train + self.output_test + self.pred_train + self.pred_test

    def _matrix_ranges(self) -> Matrix:
        return matrix_ranges(self._concatenate_outputs(), nb_points=10)

    def _ref_val_names(self) -> List[str]:
        return [[name + '_ref', name + '_pred'] for name in self.output_names]

    def _tooltip(self) -> Tooltip:
        return Tooltip(self.input_names + sum(self._ref_val_names(), []))

    def _points(self, inputs: Matrix, ref_outputs: Matrix, pred_outputs: Matrix) -> Points:
        points_list = []
        for input_, ref_out, pred_out in zip(inputs, ref_outputs, pred_outputs):
            points_list.append({attr: input_[col] for col, attr in enumerate(self.input_names)})
            points_list[-1].update({f"{attr}_ref": ref_out[col] for col, attr in enumerate(self.output_names)})
            points_list[-1].update({f"{attr}_pred": pred_out[col] for col, attr in enumerate(self.output_names)})
        return points_list

    def _ref_val_datasets(self, points_train: Points, points_test: Points) -> List[pl_Dataset]:
        tooltip = self._tooltip()
        ref_args = {'point_style': REF_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Train data', 'tooltip': tooltip}
        val_args = {'point_style': VAL_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Test data', 'tooltip': tooltip}
        return [pl_Dataset(elements=points_test, **val_args), pl_Dataset(elements=points_train, **ref_args)]

    def _bisectrice_points(self) -> Points:
        hack_bisectrices = []
        for point in zip(*self._matrix_ranges()):
            hack_bisectrices.append({f"{self.output_names[0]}_ref": point[0], f"{self.output_names[0]}_pred": point[0]})
            for idx, name in enumerate(self.output_names):
                hack_bisectrices[-1].update({name + '_ref': point[idx], name + '_pred': point[idx]})
        return hack_bisectrices

    def _to_val_points(self) -> List[pl_Dataset]:
        points_train = self._points(self.input_train, self.output_train, self.pred_train)
        points_test = self._points(self.input_test, self.output_test, self.pred_test)
        return points_train, points_test, self._bisectrice_points()

    def build_graphs(self) -> List[Graph2D]:
        """
        Build elements and graphs for plot_data method.
        """
        points_train, points_test, points_bisectrice = self._to_val_points()
        pl_datasets = self._ref_val_datasets(points_train, points_test)
        pl_datasets.append(pl_Dataset(points_bisectrice, point_style=LIN_POINT_STYLE, edge_style=STD_LINE,
                                      name="Reference = Predicted"))

        graphs = []
        for (ref, pred) in self._ref_val_names():
            graphs.append(Graph2D(graphs=pl_datasets, axis=axis_style(10, 10), x_variable=ref, y_variable=pred))
        return graphs, points_train + points_test + points_bisectrice

    def plot_data(self, **_):
        """
        Plot data method for ValidationData.
        """
        graphs, elements = self.build_graphs()
        return [MultiplePlots(elements=elements, plots=graphs, initial_view_on=True)]


class ModelValidation(DessiaObject):
    _non_data_eq_attributes = ['_score']

    def __init__(self, modeler: Modeler, validation_data: ValidationData, name: str = ''):
        self.modeler = modeler
        self.data = validation_data
        self._score = None
        DessiaObject.__init__(self, name=name)
# TODO: is this too heavy ?

    @property
    def score(self) -> float:
        """
        Score of fitted Modeler stored in attribute modeler.
        """
        if self._score is None:
            self._score = self.modeler.score_matrix(self.data.input_test, self.data.output_test)
        return self._score

    @classmethod
    def _build(cls, modeler: Modeler, input_train: Matrix, input_test: Matrix, output_train: Matrix,
               output_test: Matrix, input_names: List[str], output_names: List[str], ratio: float = 0.8,
               name: str = '') -> 'ModelValidation':
        trained_mdlr, pred_test = Modeler.fit_predict_matrix(input_train, output_train, input_test, modeler.model,
                                                             modeler.in_scaled, modeler.out_scaled, name)
        pred_train = trained_mdlr.predict_matrix(input_train)
        validation_data = ValidationData(input_train, input_test, output_train, output_test, pred_train, pred_test,
                                         input_names, output_names, f"{name}_data")
        return cls(trained_mdlr, validation_data, name)


    @classmethod
    def from_matrix(cls, modeler: Modeler, inputs: Matrix, outputs: Matrix, input_names: List[str],
                    output_names: List[str], ratio: float = 0.8, name: str = '') -> 'ModelValidation':
        """
        Create a ModelValidation object from inputs and outputs matrices.

        :param modeler:
            Modeler type and its hyperparameters, stored in a Modeler object for the sake of simplicity. Here, modeler
             does not need to be fitted.
        :type modeler: Modeler

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param input_names:
            Names of input features
        :type input_names: List[str]

        :param output_names:
            Names of output features
        :type output_names: List[str]

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, ind_train will be of length `int(ratio)` and ind_test of
            length `len_matrix - int(ratio)`.
        :type ratio: float, `optional`, defaults to 0.8

        :param name:
            Name of ModelValidation
        :type name: str, `optional`, defaults to `''`

        :return: A ModelValidation object, containing the fitted modeler, its score, train and test data and their
         predictions for input, stored in a ValidationData object.
        :rtype: ModelValidation
        """
        in_train, in_test, out_train, out_test = models.train_test_split(inputs, outputs, ratio=ratio)
        return cls._build(modeler, in_train, in_test, out_train, out_test, input_names, output_names, ratio, name)

    @classmethod
    def from_dataset(cls, modeler: Modeler, dataset: Dataset, input_names: List[str], output_names: List[str],
                     ratio: float = 0.8, name: str = '') -> 'ModelValidation':
        """
        Create a ModelValidation object from a dataset.

        :param modeler:
            Modeler type and its hyperparameters, stored in a Modeler object for the sake of simplicity. Here, modeler
             does not need to be fitted.
        :type modeler: Modeler

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features
        :type input_names: List[str]

        :param output_names:
            Names of output features
        :type output_names: List[str]

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, ind_train will be of length `int(ratio)` and ind_test of
            length `len_matrix - int(ratio)`.
        :type ratio: float, `optional`, defaults to 0.8

        :param name:
            Name of ModelValidation
        :type name: str, `optional`, defaults to `''`

        :return: A ModelValidation object, containing the fitted modeler, its score, train and test data and their
         predictions for input, stored in a ValidationData object.
        :rtype: ModelValidation
        """
        in_train, in_test, out_train, out_test = dataset.train_test_split(input_names, output_names, ratio)
        return cls._build(modeler, in_train, in_test, out_train, out_test, input_names, output_names, ratio, name)

    def plot_data(self, **_):
        """
        Plot data method for ModelValidation.
        """
        return self.data.plot_data()


class CrossValidation(DessiaObject):
    """
    Class to cross validate a Modeler modelisation.

    The purpose of cross validation is to validate a modelisation process for a specific type of machine learning
    method, set with specific hyperparameters.
    The first step of cross validation is to split data into train and test data. Then the model is fitted with
    train data and scored with test data. Furthermore, train and test inputs are predicted with the model and
    plotted in a graph that plots these predictions versus reference values. In this plot, the more red points are
    near the black line, the more the model can predict new data precisely.
    This process of cross validation is ran nb_tests times. If all of them show a good score and a nice train test
    plot, then the tested modelisation is validated and can be used in other, but similar, processes for
    predictions.
    """
    _non_data_eq_attributes = ['_scores']

    def __init__(self, model_validations: List[ModelValidation], name: str = ''):
        self.model_validations = model_validations
        self._scores = None
        DessiaObject.__init__(self, name=name)

    @property
    def scores(self) -> Vector:
        """
        List of scores of modelers contained in model_validations.
        """
        if self._scores is None:
            self._scores = [model_val.score for model_val in self.model_validations]
            return self._scores
        return self._scores

    @property
    def _points_scores(self) -> Points:
        scores = self.scores
        points_scores = []
        for idx, score in enumerate(scores):
            points_scores.append({'Index': idx, 'Score': score})
        return points_scores

    def _plot_score(self) -> Graph2D:
        scores = self._points_scores
        nidx = len(scores)
        limits = pl_Dataset(elements=scores_limits(nidx), point_style=INV_POINT_STYLE, edge_style=NO_LINE)
        axis = axis_style(nidx, nidx)

        scores_ds = pl_Dataset(elements=scores, tooltip=Tooltip(['Index', 'Score']), point_style=REF_POINT_STYLE,
                               edge_style=STD_LINE, name="Scores")

        return Graph2D(x_variable='Index', y_variable='Score', graphs=[scores_ds, limits], axis=axis)

    @classmethod
    def from_matrix(cls, modeler: Modeler, inputs: Matrix, outputs: Matrix, input_names: List[str],
                    output_names: List[str], nb_tests: int = 5, ratio: float = 0.8,
                    name: str = '') -> 'CrossValidation':
        """
        Cross Validation of modeler from inputs and outputs matrices, given input_names and output_names.
        """
        validations = []
        for idx in range(nb_tests):
            name = f"{name}_val_{idx}"
            validations.append(ModelValidation.from_matrix(modeler, inputs, outputs, input_names, output_names, ratio,
                                                           name))
        return cls(validations, name)

    @classmethod
    def from_dataset(cls, modeler: Modeler, dataset: Dataset, input_names: List[str], output_names: List[str],
                     nb_tests: int = 5, ratio: float = 0.8) -> 'CrossValidation':
        """
        Cross Validation of modeler from a Dataset object, given input_names and output_names.

        :param modeler:
            Modeler type and its hyperparameters, stored in a Modeler object for the sake of simplicity. Here, modeler
             does not need to be fitted.
        :type modeler: List[List[float]]

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features
        :type input_names: List[str]

        :param output_names:
            Names of output features
        :type output_names: List[str]

        :param nb_tests:
            Number of train test validation to run in cross_validation method
        :type nb_tests: int, `optional`, defaults to 1

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, ind_train will be of length `int(ratio)` and ind_test of
            length `len_matrix - int(ratio)`.
        :type ratio: float, `optional`, defaults to 0.8
        """
        validations = []
        for idx in range(nb_tests):
            name = f"{modeler.name}_val_{idx}"
            validations.append(ModelValidation.from_dataset(modeler, dataset, input_names, output_names, ratio, name))
        return cls(validations, f"{name}_crossval")

    def plot_data(self, **_):
        """
        Plot data method for CrossValidation.
        """
        graphs = []
        for idx, validation in enumerate(self.model_validations):
            graphs += validation.data.build_graphs()[0]
        scores_graph = [self._plot_score()]
        return scores_graph + [MultiplePlots(elements=[{"factice_key":0}], plots=graphs, initial_view_on=True)]



def matrix_ranges(matrix: Matrix, nb_points: int = 20) -> Matrix:
    """
    Dessia linspace of nb_points points between extremum of each column of matrix.
    """
    ranges = []
    for feature_column in zip(*matrix):
        min_value = min(feature_column)
        max_value = max(feature_column)
        step_range = (max_value - min_value)/nb_points
        ranges.append(npy.arange(min_value, max_value, step_range).tolist() + [1.05 * max_value])
    return ranges

def axis_style(nb_x: int = 10, nb_y: int = 10) -> Axis:
    """
    Set axis style for Modeler objects.
    """
    return Axis(nb_points_x=nb_x, nb_points_y=nb_y, axis_style=STD_LINE, grid_on=True)

def scores_limits(number: int) -> Points:
    """
    Draw white points in scatter for it to be plotted between 0 and number on x axis and 0 and 1 on y axis.
    """
    return [{'Index': -0.05, 'Score': -0.05}, {'Index': number + 0.05, 'Score': 1.05}]
