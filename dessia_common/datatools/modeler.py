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
    def _pl_samples(inputs: Matrix, ref_outputs: Matrix, pred_outputs: Matrix, input_names: List[str],
                    output_names: List[str]) -> List[Dict[str, float]]:
        plot_data_list = []
        for input_, ref_output, pred_output in zip(inputs, ref_outputs, pred_outputs):
            plot_data_list.append({attr: input_[col] for col, attr in enumerate(input_names)})
            plot_data_list[-1].update({attr + '_ref': ref_output[col] for col, attr in enumerate(output_names)})
            plot_data_list[-1].update({attr + '_pred': pred_output[col] for col, attr in enumerate(output_names)})
        return plot_data_list

    def _graph_dataset(self, pl_samples: List[Dict[str, Any]], **kwargs) -> pl_Dataset:
        return pl_Dataset(elements=pl_samples, **kwargs)

    @staticmethod
    def _ref_pred_names(input_names: List[str], output_names: List[str]) -> Tooltip:
        return [[name + '_ref', name + '_pred'] for name in output_names]

    def _ref_pred_datasets(self, pl_samples_train: List[Dict[str, float]], pl_samples_test: List[Dict[str, float]],
                           tooltip: Tooltip) -> List[pl_Dataset]:
        ref_args = {'point_style': REF_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Train data', 'tooltip': tooltip}
        val_args = {'point_style': VAL_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Test data', 'tooltip': tooltip}
        ref_dataset = self._graph_dataset(pl_samples_train, **ref_args)
        val_dataset = self._graph_dataset(pl_samples_test, **val_args)
        return [ref_dataset, val_dataset]

    @staticmethod
    def _hack_bisectrice(output_ranges: List[Vector], output_names: List[str]) -> List[Dict[str, float]]:
        hack_bisectrices = []
        for point in zip(*output_ranges):
            hack_bisectrices.append({output_names[0] + '_ref': point[0], output_names[0] + '_pred': point[0]})
            for idx, name in enumerate(output_names):
                hack_bisectrices[-1].update({name + '_ref': point[idx], name + '_pred': point[idx]})
        return hack_bisectrices

    @staticmethod
    def _limit_dataset(output_ranges: List[Vector], output_names: List[str]) -> pl_Dataset:
        points = Modeler._hack_bisectrice(output_ranges, output_names)
        return pl_Dataset(points, point_style=LIN_POINT_STYLE, edge_style=STD_LINE, name="Reference = Predicted")

    @staticmethod
    def _train_test_dataset(dataset: Dataset, input_names: List[str], output_names: List[str],
                            ratio: float = 0.8) -> List[Matrix]:
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return models.train_test_split(inputs, outputs, ratio=ratio)

    def _to_points(self, inputs_train: Matrix, inputs_test: Matrix, outputs_train: Matrix, outputs_test: Matrix,
                           input_names: List[str], output_names: List[str]) -> List[pl_Dataset]:
        score = self._score(inputs_test, outputs_test)

        pred_inputs_train = self._predict(inputs_train)
        pred_inputs_test = self._predict(inputs_test)

        pl_samples_train = self._pl_samples(inputs_train, outputs_train, pred_inputs_train, input_names, output_names)
        pl_samples_test = self._pl_samples(inputs_test, outputs_test, pred_inputs_test, input_names, output_names)
        output_ranges = matrix_ranges(outputs_train + outputs_test + pred_inputs_train + pred_inputs_test, nb_points=10)
        hack_dataset = Modeler._limit_dataset(output_ranges, output_names)
        return pl_samples_train, pl_samples_test, hack_dataset, score

    @classmethod
    def from_dataset_fit_validate(cls, dataset: Dataset, input_names: List[str], output_names: List[str],
                                  class_: Type, hyperparameters: Dict[str, Any], input_is_scaled: bool = True,
                                  output_is_scaled: bool = False, ratio: float = 0.8, name: str = '') -> 'Modeler':

        inputs_train, inputs_test, outputs_train, outputs_test = cls._train_test_dataset(dataset, input_names,
                                                                                         output_names, ratio)

        modeler = cls._fit(inputs_train, outputs_train, class_, hyperparameters, input_is_scaled, output_is_scaled)

        pl_samples_train, pl_samples_test, hack_dataset, score = modeler._to_points(inputs_train, inputs_test,
                                                                                    outputs_train, outputs_test,
                                                                                    input_names, output_names)

        ref_pred_names = cls._ref_pred_names(input_names, output_names)
        pl_datasets = modeler._ref_pred_datasets(pl_samples_train, pl_samples_test, tooltip=Tooltip(ref_pred_names))
        pl_datasets.append(hack_dataset)

        graphs2D = []
        for (ref, pred) in ref_pred_names:
            graphs2D.append(Graph2D(graphs=pl_datasets, axis=axis_style(10, 10), x_variable=ref, y_variable=pred))

        return modeler, graphs2D, score

    @classmethod
    def cross_validation(cls, dataset: Dataset, input_names: List[str], output_names: List[str],
                         class_: Type, hyperparameters: Dict[str, Any], input_is_scaled: bool = True,
                         output_is_scaled: bool = False, nb_tests: int = 1, ratio: float = 0.8,
                         name: str = '') -> 'Modeler':
        scores = []
        all_graphs = []
        for idx in range(nb_tests):
            modeler, graphs, score = cls.from_dataset_fit_validate(dataset, input_names, output_names, class_,
                                                                   hyperparameters, input_is_scaled, output_is_scaled,
                                                                   ratio, name)
            scores.append({'Index': idx, 'Score': score})
            all_graphs += graphs
        return scores, all_graphs

    @staticmethod
    def _plot_score(scores: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[PointFamily]]:
        nidx = len(scores)
        limits = pl_Dataset(elements=scores_limits(nidx), point_style=INV_POINT_STYLE, edge_style=NO_LINE)

        scores_ds = pl_Dataset(elements=scores, tooltip=Tooltip(['Index', 'Score']), point_style=REF_POINT_STYLE,
                               edge_style=STD_LINE, name="Scores")

        return Graph2D(x_variable='Index', y_variable='Score', graphs=[scores_ds, limits], axis=axis_style(nidx, nidx))

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
        return

    def features_mrmr(self):
        return


    # def _ref_val_predict(self, ref_inputs: Matrix, val_inputs: Matrix) -> Tuple[Matrix]:
    #     scaled_ref_predictions = self._predict(ref_inputs)
    #     scaled_val_predictions = self._predict(val_inputs)
    #     return self.output_scaler.inverse_transform_matrices(scaled_ref_predictions, scaled_val_predictions)



    # @staticmethod
    # def _hack_bisectrice(matrix: Matrix, output_names: List[str]):
    #     output_ranges = matrix_ranges(matrix)
    #     hack_bisectrices = []
    #     for point in zip(*output_ranges):
    #         hack_bisectrices.append({output_names[0] + '_ref': point[0], output_names[0] + '_pred': point[0]})
    #         for idx, name in enumerate(output_names):
    #             hack_bisectrices[-1].update({name + '_ref': point[idx], name + '_pred': point[idx]})
    #     return hack_bisectrices

    # def _limit_dataset(self, matrix: Matrix, output_names: List[str], name: str) -> pl_Dataset:
    #     hak_scatter = self._hack_bisectrice(matrix, output_names)
    #     return pl_Dataset(elements=hak_scatter, point_style=LIN_POINT_STYLE, edge_style=STD_LINE, name=name)

    # def _ref_val_plot(self, pl_samples: List[Dict[str, Any]], ref_inputs: Matrix, val_inputs: Matrix,
    #                   input_names: List[str], tooltip: Tooltip) -> List[pl_Dataset]:
    #     ref_args = {'point_style': REF_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Train data', 'tooltip': tooltip}
    #     val_args = {'point_style': VAL_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Test data', 'tooltip': tooltip}

    #     ref_predictions, val_predictions = self._ref_val_predict(ref_inputs, val_inputs)
    #     ref_plot = self._graph_dataset(pl_samples, **ref_args)
    #     val_plot = self._graph_dataset(pl_samples, **val_args)
    #     hak_plot = self._limit_dataset(pl_samples, "Reference = Predicted")
    #     return [ref_plot, val_plot, hak_plot]

    # def _validation_graphs(self, pl_samples: List[Dict[str, Any]], ref_inputs: Matrix, val_inputs: Matrix,
    #                        input_names: List[str], output_names: List[str]) -> List[Graph2D]:
    #     refpred_names = [[name + '_ref', name + '_pred'] for name in output_names]
    #     tooltip = Tooltip(input_names + sum(refpred_names, []))
    #     all_graphs = self._ref_val_plot(pl_samples, ref_inputs, val_inputs, input_names, tooltip)

    #     graphs2D = []
    #     for (ref, pred) in refpred_names:
    #         graphs2D.append(Graph2D(graphs=all_graphs, axis=axis_style(10, 10), x_variable=ref, y_variable=pred))
    #     return graphs2D

    # @classmethod # TODO: Make code nice
    # def _dataset_fit_validate(cls, dataset: Dataset, input_names: List[str], output_names: List[str], class_: Type,
    #                           hyperparameters: Dict[str, Any], input_is_scaled: bool = True,
    #                           output_is_scaled: bool = False, ratio: float = 0.8, name: str = '') -> 'Modeler':
    #     inputs = dataset.sub_matrix(input_names)
    #     outputs = dataset.sub_matrix(output_names)
    #     inputs_train, inputs_test, outputs_train, outputs_test = models.train_test_split(inputs, outputs, ratio=ratio)

    #     modeler = cls._fit(inputs_train, outputs_train, class_, hyperparameters, input_is_scaled, output_is_scaled)

    #     graphs = modeler._validation_graphs(inputs_train, inputs_test, input_names, output_names)

    #     return modeler, graphs, modeler._score(inputs_test, outputs_test)


    # def _validation_plot(self, ref_inputs: Matrix, ref_outputs: Matrix, val_inputs: Matrix, val_outputs: Matrix,
    #                      input_names: List[str], output_names: List[str]):
    #     graphs = self._validation_graphs(ref_inputs, ref_outputs, val_inputs, val_outputs, input_names, output_names)
    #     return MultiplePlots(plots=graphs, initial_view_on=True)





def min_max_range(inf_borns: List[float], sup_borns: List[float], keys: List[str], nb_points: int = 10) -> List[Vector]:
    ranges = []
    for inf_born, sup_born in zip(inf_borns, sup_borns):
        step_range = (sup_born - inf_born)/nb_points
        ranges.append(npy.arange(inf_born, sup_born, step_range).tolist())
    return ranges

def matrix_ranges(matrix: Matrix, nb_points: int = 20) -> List[Vector]:
    matrix_ranges = []
    for feature_column in zip(*matrix):
        min_value = min(feature_column)
        max_value = max(feature_column)
        step_range = (max_value - min_value)/nb_points
        matrix_ranges.append(npy.arange(min_value, max_value, step_range).tolist())
    return matrix_ranges

def axis_style(nb_x: int = 10, nb_y: int = 10) -> Axis:
    return Axis(nb_points_x=nb_x, nb_points_y=nb_y, axis_style=STD_LINE, grid_on=True)

def scores_limits(number: int) -> List[Dict[str, Any]]:
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



