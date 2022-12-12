"""
Librairy for building machine learning modelers from Dataset or Lists using sklearn models handled in models.
"""
from typing import List, Dict, Any, Tuple, Union, Type

import numpy as npy

from plot_data.core import Dataset as pl_Dataset
from plot_data.core import Line2D, Scatter, EdgeStyle, PointFamily, Tooltip, MultiplePlots, PrimitiveGroup, PointStyle, Graph2D
from plot_data.colors import BLACK, RED, BLUE

from dessia_common.core import DessiaObject
from dessia_common.datatools.dataset import Dataset
import dessia_common.datatools.models as models


Vector = List[float]
Matrix = List[Vector]

class Modeler(DessiaObject):
    def __init__(self, model: models.Model, input_scaler: models.Scaler, output_scaler: models.Scaler, name: str = ''):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        DessiaObject.__init__(self, name=name)

    @staticmethod
    def _set_scaler_class(is_scaled: bool) -> models.Scaler:
        if is_scaled:
            return models.StandardScaler
        return models.IdentityScaler

    @staticmethod
    def _set_scaler_name(modeler_name: str, in_out: str, is_scaled: bool) -> str:
        name = f"{modeler_name}_"
        return name + (f"{in_out}_scaler" if is_scaled else "indentity_scaler")

    @staticmethod
    def _set_scaler(modeler_name: str, in_out: str, is_scaled: bool) -> models.Scaler:
        class_ = Modeler._set_scaler_class(is_scaled)
        name = Modeler._set_scaler_name(modeler_name, in_out, is_scaled)
        return class_, name

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
        in_scaler_class, input_scaler_name = cls._set_scaler(name, "in", input_is_scaled)
        out_scaler_class, output_scaler_name = cls._set_scaler(name, "out", output_is_scaled)

        in_scaler, scaled_inputs = in_scaler_class.fit_transform(inputs, input_scaler_name)
        out_scaler, scaled_outputs = out_scaler_class.fit_transform(outputs, output_scaler_name)

        model = class_.fit(scaled_inputs, scaled_outputs, **hyperparameters, name=name + '_model')
        return cls(model=model, input_scaler=in_scaler, output_scaler=out_scaler, name=name)

    def predict_matrix(self, inputs: List[List[float]]) -> Matrix: # TODO check type Vector or Matrix. Must be handled in Modeler.
        """
        Method to predict outputs from inputs with the current Modeler for matrix data.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`
        :type inputs: List[List[float]]

        :return: The predicted values for inputs.
        :rtype: List[List[float]]
        """
        scaled_inputs = self.input_scaler.transform(inputs)
        scaled_outputs = self.model.predict(scaled_inputs)
        return self.output_scaler.inverse_transform(scaled_outputs)

    @classmethod
    def fit_predict_matrix(cls, inputs: Matrix, outputs: Matrix, predicted_outputs: Matrix, class_: Type,
                    hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                    name: str = '') -> Tuple['Modeler', Matrix]: # TODO check type Vector or Matrix. Must be handled in Modeler.
        """
        Fit outputs to inputs and predict outputs of predicted_inputs for matrix data (fit then predict).
        """
        modeler = cls.fit_matrix(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)
        return modeler, modeler.predict_matrix(predicted_outputs)

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
        inputs = dataset.sub_matrix(input_names)
        outputs = dataset.sub_matrix(output_names)
        return cls.fit_matrix(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)

    def predict_dataset(self, dataset: Dataset, input_names: List[str]) -> Matrix: # TODO check type Vector or Matrix. Must be handled in Modeler.
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
    def fit_predict_dataset(cls, dataset: Dataset, input_names: List[str], output_names: List[str],
                            predicted_names: List[str], class_: Type, hyperparameters: Dict[str, Any],
                            input_is_scaled: bool = True, output_is_scaled: bool = False,
                            name: str = '') -> Tuple['Modeler', Matrix]: # TODO check type Vector or Matrix. Must be handled in Modeler.
        """
        Fit outputs to inputs and predict outputs of predicted_inputs for Dataset object (fit then predict).
        """
        modeler = cls.fit_dataset(dataset, input_names, output_names, class_, hyperparameters, input_is_scaled,
                                  output_is_scaled, name)
        return modeler, modeler.predict_dataset(predicted_names)

    @staticmethod
    def _hack_bisectrice(ref_outputs: Matrix, val_outputs: Matrix, output_names: List[str]):
        output_ranges = []
        for ref_output, val_output in zip(zip(*ref_outputs), zip(*val_outputs)):
            min_value = min(min(ref_output), min(val_output))
            max_value = max(max(ref_output), max(val_output))
            step_range = (max_value - min_value)/20
            output_ranges.append(npy.arange(min_value, max_value, step_range).tolist())

        bisectrices_points = list(zip(*output_ranges))
        hack_bisectrices = []
        for point in bisectrices_points:
            hack_bisectrices.append({output_names[0] + '_ref': point[0], output_names[0] + '_pred': point[0]})
            for idx, name in enumerate(output_names):
                hack_bisectrices[-1].update({name + '_ref': point[idx], name + '_pred': point[idx]})
        return hack_bisectrices

    def _plot_data_list(self, inputs: Matrix, ref_outputs: Matrix, pred_outputs: Matrix, input_names: List[str],
                        output_names: List[str]):
        plot_data_list = []
        for input_, ref_output, pred_output in zip(inputs, ref_outputs, pred_outputs):
            plot_data_list.append({attr: input_[col] for col, attr in enumerate(input_names)})
            plot_data_list[-1].update({attr + '_ref': ref_output[col] for col, attr in enumerate(output_names)})
            plot_data_list[-1].update({attr + '_pred': pred_output[col] for col, attr in enumerate(output_names)})
        return plot_data_list

    def _validation_plot(self, ref_inputs: Matrix, ref_outputs: Matrix, val_inputs: Matrix, val_outputs: Matrix,
                         input_names: List[str], output_names: List[str]):
        ref_predictions = self.model.predict(ref_inputs)
        val_predictions = self.model.predict(val_inputs)

        ref_scatter = self._plot_data_list(ref_inputs, ref_outputs, ref_predictions, input_names, output_names)
        val_scatter = self._plot_data_list(val_inputs, val_outputs, val_predictions, input_names, output_names)
        hak_scatter = self._hack_bisectrice(ref_outputs, val_outputs, output_names)

        tooltip_attributes = input_names + sum([[output_name + '_ref', output_name + '_pred']
                                                for output_name in output_names], [])

        ref_graph = pl_Dataset(elements=ref_scatter,
                               point_style=PointStyle(BLUE, BLUE, 0.1, 2., 'circle'),
                               edge_style=EdgeStyle(0.0001),
                               name="Reference predictions", tooltip=Tooltip(tooltip_attributes))

        val_graph = pl_Dataset(elements=val_scatter,
                               point_style=PointStyle(RED, RED, 0.1, 2., 'circle'),
                               edge_style=EdgeStyle(0.0001),
                               name="Validation predictions", tooltip=Tooltip(tooltip_attributes))

        hak_graph = pl_Dataset(elements=hak_scatter,
                               point_style=PointStyle(BLACK, BLACK, 0.1, 1, 'crux'),
                               edge_style=EdgeStyle(1.,color_stroke=BLACK),
                               name="Bisectrice", tooltip=Tooltip(tooltip_attributes))

        graphs2D = []
        for idx, name in enumerate(output_names):
            graphs2D.append(Graph2D(graphs=[ref_graph, val_graph, hak_graph],
                                    x_variable=name + '_ref', y_variable=name + '_pred'))

        return MultiplePlots(plots=graphs2D, initial_view_on=True)

    def plot_data(self, ref_inputs: Matrix = None, ref_outputs: Matrix = None, val_inputs: Matrix = None,
                  val_outputs: Matrix = None, input_names: List[str] = None, output_names: List[str] = None):
        return [self._validation_plot(ref_inputs, ref_outputs, val_inputs, val_outputs, input_names, output_names)]


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




