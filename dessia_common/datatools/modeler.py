"""
Librairy for building machine learning modelers from Dataset or Lists using sklearn models handled in models.
"""
from typing import List, Dict, Any, Tuple, Type, Union

import numpy as npy

from plot_data.core import Dataset as pl_Dataset
from plot_data.core import EdgeStyle, Tooltip, MultiplePlots, PointStyle, Graph2D, Axis, PointFamily
from plot_data.colors import BLACK, RED, BLUE, WHITE

from dessia_common.core import DessiaObject
from dessia_common.datatools.dataset import Dataset
import dessia_common.datatools.models as models


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
    def __init__(self, model: models.Model, input_scaler: models.Scaler, output_scaler: models.Scaler, name: str = ''):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        DessiaObject.__init__(self, name=name)

    def _format_output(self, scaled_outputs: Matrix):
        """
        Format output to List[List[float]] in any case for code consistency and simplicity.
        """
        if not isinstance(scaled_outputs, list):
            return [[value] for value in self.output_scaler.inverse_transform(scaled_outputs)]
        return self.output_scaler.inverse_transform(scaled_outputs)

    @classmethod
    def _fit(cls, inputs: Matrix, outputs: Matrix, class_: Type, hyperparameters: Dict[str, Any],
             input_is_scaled: bool = True, output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Private method to fit outputs to inputs with a machine learning method from datatools.models objects.
        """
        in_scaler_class, input_scaler_name = models.Scaler.set_in_modeler(name, "in", input_is_scaled)
        out_scaler_class, output_scaler_name = models.Scaler.set_in_modeler(name, "out", output_is_scaled)

        in_scaler, scaled_inputs = in_scaler_class.fit_transform(inputs, input_scaler_name)
        out_scaler, scaled_outputs = out_scaler_class.fit_transform(outputs, output_scaler_name)

        model = class_.fit(scaled_inputs, scaled_outputs, **hyperparameters, name=name + '_model')
        return cls(model=model, input_scaler=in_scaler, output_scaler=out_scaler, name=name)

    @classmethod
    def fit_matrix(cls, inputs: Matrix, outputs: Matrix, class_: Type, hyperparameters: Dict[str, Any],
                   input_is_scaled: bool = True, output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Method to fit outputs to inputs with a machine learning method from datatools.models objects for matrix data.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`
        :type outputs: List[List[float]]

        :param class_:
            Class of datatools.models objetc to use for fitting, e.g. RandomForestRegressor, LinearRegression,...
        :type class_: Type

        :param input_is_scaled:
            Whether to standardize inputs or not with a models.StandardScaler
        :type input_is_scaled: bool, `optional`, True

        :param output_is_scaled:
            Whether to standardize outputs or not with a models.StandardScaler
        :type output_is_scaled: bool, `optional`, False

        :param hyperparameters:
            Hyperparameters of the used scikit-learn object.
        :type hyperparameters: dict[str, Any], `optional`

        :param name:
            Name of Model
        :type name: str, `optional`, defaults to `''`

        :return: The equivalent Modeler object containing the fitted model and scalers associated to inputs and outputs
        :rtype: Modeler
        """
        return cls._fit(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)

    @classmethod
    def fit_dataset(cls, dataset: Dataset, input_names: List[str], output_names: List[str], class_: Type,
                    hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                    name: str = '') -> 'Modeler':
        """
        Method to fit outputs to inputs with a machine learning method from datatools.models objects for a Dataset.

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features
        :type inputs: List[str]

        :param output_names:
            Names of output features
        :type inputs: List[str]

        :param class_:
            Class of datatools.models objetc to use for fitting, e.g. RandomForestRegressor, LinearRegression,...
        :type class_: Type

        :param hyperparameters:
            Hyperparameters of the used scikit-learn object.
        :type hyperparameters: dict[str, Any], `optional`

        :param input_is_scaled:
            Whether to standardize inputs or not with a models.StandardScaler
        :type input_is_scaled: bool, `optional`, True

        :param output_is_scaled:
            Whether to standardize outputs or not with a models.StandardScaler
        :type output_is_scaled: bool, `optional`, False

        :param name:
            Name of Model
        :type name: str, `optional`, defaults to `''`

        :return: The equivalent Modeler object containing the fitted model and scalers associated to inputs and outputs
        :rtype: Modeler
        """
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return cls.fit_matrix(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)

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
        :type inputs: List[str]

        :return: The predicted values for inputs.
        :rtype: List[List[float]]
        """
        inputs = dataset.sub_matrix(input_names)
        return self.predict_matrix(inputs)

    @classmethod
    def _fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_outputs: Matrix, class_: Type,
                     hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                     name: str = '') -> Tuple['Modeler', Union[Vector, Matrix]]:
        """
        Private method to fit outputs to inputs and predict predicted_outputs for a Dataset (fit then predict).
        """
        modeler = cls._fit(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)
        return modeler, modeler._predict(predicted_outputs)

    @classmethod
    def fit_predict_matrix(cls, inputs: Matrix, outputs: Matrix, predicted_outputs: Matrix, class_: Type,
                           hyperparameters: Dict[str, Any], input_is_scaled: bool = True,
                           output_is_scaled: bool = False, name: str = '') -> Tuple['Modeler', Matrix]:
        """
        Fit outputs to inputs and predict predicted_outputs for matrix data (fit then predict).
        """
        modeler, predictions = cls._fit_predict(inputs, outputs, predicted_outputs, class_, hyperparameters,
                                                input_is_scaled, output_is_scaled, name)
        return modeler, cls._format_output(predictions)

    @classmethod
    def fit_predict_dataset(cls, fit_dataset: Dataset, to_predict_dataset: Dataset, input_names: List[str],
                            output_names: List[str], class_: Type, hyperparameters: Dict[str, Any],
                            input_is_scaled: bool = True, output_is_scaled: bool = False,
                            name: str = '') -> Tuple['Modeler', Matrix]:
        """
        Fit outputs to inputs and predict outputs of to_predict_dataset (fit then predict).
        """
        modeler = cls.fit_dataset(fit_dataset, input_names, output_names, class_, hyperparameters, input_is_scaled,
                                  output_is_scaled, name)
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
        :type inputs: List[str]

        :param output_names:
            Names of output features
        :type inputs: List[str]

        :return: The score of Modeler.
        :rtype: float
        """
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return self._score(inputs, outputs)

    @staticmethod
    def _pl_points(inputs: Matrix, ref_outputs: Matrix, pred_outputs: Matrix, input_names: List[str],
                   output_names: List[str]) -> Points:
        plot_data_list = []
        for input_, ref_output, pred_output in zip(inputs, ref_outputs, pred_outputs):
            plot_data_list.append({attr: input_[col] for col, attr in enumerate(input_names)})
            plot_data_list[-1].update({attr + '_ref': ref_output[col] for col, attr in enumerate(output_names)})
            plot_data_list[-1].update({attr + '_pred': pred_output[col] for col, attr in enumerate(output_names)})
        return plot_data_list

    @staticmethod
    def _plot_dataset(pl_points: Points, **kwargs) -> pl_Dataset:
        return pl_Dataset(elements=pl_points, **kwargs)

    @staticmethod
    def _ref_val_names(input_names: List[str], output_names: List[str]) -> Tooltip:
        return [[name + '_ref', name + '_pred'] for name in output_names]

    @staticmethod
    def _ref_val_datasets(pl_points_train: Points, pl_points_test: Points, tooltip: Tooltip) -> List[pl_Dataset]:
        ref_args = {'point_style': REF_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Train data', 'tooltip': tooltip}
        val_args = {'point_style': VAL_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Test data', 'tooltip': tooltip}
        ref_dataset = Modeler._plot_dataset(pl_points_train, **ref_args)
        val_dataset = Modeler._plot_dataset(pl_points_test, **val_args)
        return [ref_dataset, val_dataset]

    @staticmethod
    def _hack_bisectrice(output_ranges: Matrix, output_names: List[str]) -> Points:
        hack_bisectrices = []
        for point in zip(*output_ranges):
            hack_bisectrices.append({output_names[0] + '_ref': point[0], output_names[0] + '_pred': point[0]})
            for idx, name in enumerate(output_names):
                hack_bisectrices[-1].update({name + '_ref': point[idx], name + '_pred': point[idx]})
        return hack_bisectrices

    @staticmethod
    def _bisectrice_points(output_ranges: Matrix, output_names: List[str]) -> pl_Dataset:
        points = Modeler._hack_bisectrice(output_ranges, output_names)
        return pl_Dataset(points, point_style=LIN_POINT_STYLE, edge_style=STD_LINE, name="Reference = Predicted")

    def _to_val_points(self, inputs_train: Matrix, inputs_test: Matrix, outputs_train: Matrix, outputs_test: Matrix,
                       input_names: List[str], output_names: List[str]) -> List[pl_Dataset]:

        pred_inputs_train = self._predict(inputs_train)
        pl_points_train = self._pl_points(inputs_train, outputs_train, pred_inputs_train, input_names, output_names)

        pred_inputs_test = self._predict(inputs_test)
        pl_points_test = self._pl_points(inputs_test, outputs_test, pred_inputs_test, input_names, output_names)

        output_ranges = matrix_ranges(outputs_train + outputs_test + pred_inputs_train + pred_inputs_test, nb_points=10)
        hack_dataset = Modeler._bisectrice_points(output_ranges, output_names)
        return pl_points_train, pl_points_test, hack_dataset

    def _build_graphs(self, inputs_train: Matrix, inputs_test: Matrix, outputs_train: Matrix, outputs_test: Matrix,
                      input_names: List[str], output_names: List[str]) -> List[Graph2D]:
        pl_points_train, pl_points_test, hack_dataset = self._to_val_points(inputs_train, inputs_test, outputs_train,
                                                                            outputs_test, input_names, output_names)
        ref_val_names = self._ref_val_names(input_names, output_names)
        tooltip = Tooltip(input_names + sum(ref_val_names, []))

        pl_datasets = self._ref_val_datasets(pl_points_train, pl_points_test, tooltip=tooltip)
        pl_datasets.append(hack_dataset)

        graphs2D = []
        for (ref, pred) in ref_val_names:
            graphs2D.append(Graph2D(graphs=pl_datasets, axis=axis_style(10, 10), x_variable=ref, y_variable=pred))
        return graphs2D

    @classmethod
    def _fit_score(cls, inputs_train: Matrix, inputs_test: Matrix, outputs_train: Matrix, outputs_test: Matrix,
                   input_names: List[str], output_names: List[str], class_: Type, hyperparameters: Dict[str, Any],
                   input_is_scaled: bool, output_is_scaled: bool, name: str) -> 'Modeler':
        """
        Train test split dataset, fit modeler with train matrices and test it with test matrices.
        """
        modeler = cls._fit(inputs_train, outputs_train, class_, hyperparameters, input_is_scaled, output_is_scaled)
        return modeler, modeler._score(inputs_test, outputs_test)

    @classmethod
    def from_dataset_fit_score(cls, dataset: Dataset, input_names: List[str], output_names: List[str], class_: Type,
                               hyperparameters: Dict[str, Any], input_is_scaled: bool = True,
                               output_is_scaled: bool = False, ratio: float = 0.8, name: str = '') -> 'Modeler':
        """
        Train test split dataset, fit modeler with train matrices and score it with test matrices.
        """
        train_test_matrices = dataset.train_test_split(input_names, output_names, ratio)
        return cls._fit_score(*train_test_matrices, input_names, output_names, class_, hyperparameters, input_is_scaled,
                              output_is_scaled, name)

    @classmethod
    def cross_validation(cls, dataset: Dataset, input_names: List[str], output_names: List[str], class_: Type,
                         hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                         nb_tests: int = 1, ratio: float = 0.8, name: str = ''):
        """
        Cross validation for a model of Models and its hyperparameters.

        The purpose of this method is to validate a modelisation process for a specific type of machine learning method,
        set with specific hyperparameters.
        The first step of cross validation is to split data into train and test data. Then the model is fitted with
        train data and scored with test data. Furthermore, train and test inputs are predicted with the model and
        plotted in a graph that plots these predictions versus reference values. In this plot, the more red points are
        near the black line, the more the model can predict new data precisely.
        This process of cross validation is ran nb_tests times. If all of them show a good score and a nice train test
        plot, then the tested modelisation is validated and can be used in other, but similar, processes for
        predictions.

        :param dataset:
            Dataset containing data, both inputs and outputs
        :type dataset: Dataset

        :param input_names:
            Names of input features
        :type inputs: List[str]

        :param output_names:
            Names of output features
        :type inputs: List[str]

        :param class_:
            Class of datatools.models objetc to use for fitting, e.g. RandomForestRegressor, LinearRegression,...
        :type class_: Type

        :param hyperparameters:
            Hyperparameters of the used scikit-learn object.
        :type hyperparameters: dict[str, Any], `optional`

        :param input_is_scaled:
            Whether to standardize inputs or not with a models.StandardScaler
        :type input_is_scaled: bool, `optional`, True

        :param output_is_scaled:
            Whether to standardize outputs or not with a models.StandardScaler
        :type output_is_scaled: bool, `optional`, False

        :param nb_tests:
            Number of train test validation to run in cross_validation method
        :type nb_tests: int, `optional`, defaults to 1

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, ind_train will be of length `int(ratio)` and ind_test of
            length `len_matrix - int(ratio)`.
        :type ratio: float, `optional`, defaults to 0.8

        :param name:
            Name of Model
        :type name: str, `optional`, defaults to `''`

        :return: All scores of models and all validation graphs, stored in list of dict to be handled in plot_data
        :rtype: Tuple[List[Dict[str, float]], List[Dict[str, float]]]
        """
        scores = []
        all_graphs = []
        for idx in range(nb_tests):
            train_test_matrices = dataset.train_test_split(input_names, output_names, ratio)
            modeler, score = cls._fit_score(*train_test_matrices, input_names, output_names, class_, hyperparameters,
                                            input_is_scaled, output_is_scaled, name)
            graphs2D = modeler._build_graphs(*train_test_matrices, input_names, output_names)
            scores.append({'Index': idx, 'Score': score})
            all_graphs += graphs2D
        return scores, all_graphs

    @staticmethod
    def _plot_score(scores: Points) -> Graph2D:
        nidx = len(scores)
        limits = pl_Dataset(elements=scores_limits(nidx), point_style=INV_POINT_STYLE, edge_style=NO_LINE)
        axis = axis_style(nidx, nidx)

        scores_ds = pl_Dataset(elements=scores, tooltip=Tooltip(['Index', 'Score']), point_style=REF_POINT_STYLE,
                               edge_style=STD_LINE, name="Scores")

        return Graph2D(x_variable='Index', y_variable='Score', graphs=[scores_ds, limits], axis=axis)

    def plot_data(self, dataset: Dataset, input_names: List[str], output_names: List[str], class_: Type,
                  hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                  nb_tests: int = 1, ratio: float = 0.8, name: str = ''):
        """
        Plot data method for Modeler.
        """
        scores, graphs = Modeler.cross_validation(dataset, input_names, output_names, class_, hyperparameters,
                                                  input_is_scaled, output_is_scaled, nb_tests, ratio, name)
        scatter_scores = self._plot_score(scores)
        return [scatter_scores] + [MultiplePlots(elements=scores, plots=graphs, initial_view_on=True)]

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


def matrix_ranges(matrix: Matrix, nb_points: int = 20) -> Matrix:
    """
    Dessia linspace of nb_points points between extremum of each column of matrix.
    """
    matrix_ranges = []
    for feature_column in zip(*matrix):
        min_value = min(feature_column)
        max_value = max(feature_column)
        step_range = (max_value - min_value)/nb_points
        matrix_ranges.append(npy.arange(min_value, max_value, step_range).tolist())
    return matrix_ranges

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


## KEPT FOR A FUTURE PLOT DATA THAT HANDLES LINE2D IN SCATTERS
# @staticmethod
# TODO: Plot data does not allow to draw shapes on plots...
# def _ref_pred_bisectrice(ref_outputs: Vector, val_outputs: Vector):
#     min_output_value = min(min(ref_outputs), min(val_outputs))
#     max_output_value = max(max(ref_outputs), max(val_outputs))
#     line_style = EdgeStyle(1., BLACK)
#     return Line2D([min_output_value, min_output_value], [max_output_value, max_output_value], line_style)

# def _validation_plot(self, ref_inputs: Matrix, ref_outputs: Matrix, val_inputs: Matrix, val_outputs: Matrix,
#                      input_names: List[str], output_names: List[str]):
#     ref_predictions = self.model.predict(ref_inputs)
#     val_predictions = self.model.predict(val_inputs)

#     ref_scatter = self._plot_data_list(ref_inputs, ref_outputs, ref_predictions, input_names, output_names)
#     val_scatter = self._plot_data_list(val_inputs, val_outputs, val_predictions, input_names, output_names)
#     # hak_scatter = self._hack_bisectrice(ref_outputs, val_outputs, output_names)
#     full_scatter = ref_scatter + val_scatter #+ hak_scatter

#     ref_index = list(range(len(ref_inputs)))
#     val_index = list(range(len(ref_inputs), len(full_scatter))) #- len(hak_scatter)))
#     # hak_index = list(range(len(hak_scatter), len(full_scatter)))

#     ref_family = PointFamily(BLUE, ref_index, name="Reference predictions")
#     val_family = PointFamily(RED, val_index, name="Validation predictions")
#     # hak_family = PointFamily(BLACK, hak_index, name="Bisectrice")
#     point_families = [ref_family, val_family] #, hak_family]

#     scatters = []
#     for idx, name in enumerate(output_names):
#         scatters.append(Scatter(x_variable=name + '_ref', y_variable=name + '_pred',
#                                 tooltip=Tooltip(list(full_scatter[0].keys()))))

#     multiplot = MultiplePlots(plots=scatters, elements=full_scatter, point_families=point_families,
#                               initial_view_on=True)

#     return multiplot




