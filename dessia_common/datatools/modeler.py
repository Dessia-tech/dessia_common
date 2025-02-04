""" Library for building machine learning modelers from Dataset or Lists using sklearn models handled in models. """
from typing import List, Dict, Tuple, Union, Any

import numpy as npy

try:
    from plot_data.core import Dataset as pl_Dataset
    from plot_data.core import (EdgeStyle, Tooltip, MultiplePlots, PointStyle, PrimitiveGroup, Axis, Sample, Point2D,
                                LineSegment2D, Label, Graph2D)
    from plot_data.colors import BLACK, RED, BLUE, WHITE
except ImportError:
    pass

from dessia_common.core import DessiaObject
from dessia_common.typings import JsonSerializable
from dessia_common.serialization import SerializableObject
from dessia_common.datatools import learning_models as models
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.math import Vector, Matrix

Points = list[dict[str, float]]

NO_LINE = EdgeStyle(0.0001)
STD_LINE = EdgeStyle(line_width=1.5, color_stroke=BLACK)

REF_POINT_STYLE = PointStyle(BLUE, BLUE, 0.1, 12, 'circle')
VAL_POINT_STYLE = PointStyle(RED, RED, 0.1, 12, 'circle')
LIN_POINT_STYLE = PointStyle(BLACK, BLACK, 0.1, 12, 'crux')
INV_POINT_STYLE = PointStyle(WHITE, WHITE, 0.1, 1, 'crux')


class SampleDataset(Dataset):
    """ Class allowing to plot and study data generated from a DOE and its prediction from a modeler modeling. """

    _standalone_in_db = True
    _allowed_methods = Dataset._allowed_methods + ["from_matrices", "matrix"]

    def __init__(self, dessia_objects: list[Sample] = None, input_names: list[str] = None,
                 output_names: list[str] = None, name: str = ''):
        self.input_names = input_names
        self.output_names = output_names
        Dataset.__init__(self, dessia_objects=dessia_objects, name=name)
        self._common_attributes = None
        if input_names is not None and output_names is not None:
            self._common_attributes = input_names + output_names

    def to_dict(self, use_pointers: bool = True, memo=None, path: str = '#', id_method=True,
                id_memo=None) -> JsonSerializable:
        """ Specific to_dict method. """
        dict_ = super().to_dict(use_pointers, memo=memo, path=path)
        for dessia_object in dict_['dessia_objects']:
            for attr in dessia_object['values']:
                dessia_object.pop(attr)
        return dict_

    @classmethod
    def dict_to_object(cls, dict_: JsonSerializable, force_generic: bool = False, global_dict=None,
                       pointers_memo: dict[str, Any] = None, path: str = '#') -> 'SerializableObject':
        """ Specific `dict_to_object` method. """
        dessia_objects = [Sample(obj['values'], obj['reference_path'], obj['name']) for obj in dict_['dessia_objects']]
        return cls(dessia_objects, dict_['input_names'], dict_['output_names'], dict_['name'])

    def _printed_attributes(self):
        return self.common_attributes

    @classmethod
    def from_matrices(cls, inputs: Matrix, predictions: Matrix, input_names: list[str], output_names: list[str],
                      name: str = '') -> 'SampleDataset':
        """ Build a `SampleDataset` from inputs matrix and their predictions matrix. """
        samples = []
        for index, (input_, pred) in enumerate(zip(inputs, predictions)):
            sample = {attr: input_[attr_index] for attr_index, attr in enumerate(input_names)}
            sample.update(dict(zip(output_names, pred)))
            samples.append(Sample(sample, reference_path=f"#/dessia_objects/{index}", name=f"{name}_{index}"))
        return cls(samples, input_names, output_names)

    @property
    def matrix(self) -> Matrix:
        """ Get equivalent matrix of `dessia_objects` (`len(dessia_objects) x len(common_attributes)`). """
        if self._matrix is None:
            matrix = []
            for sample in self:
                vector_features, temp_row = list(zip(*list(sample.values.items())))
                matrix.append([temp_row[vector_features.index(attr)] for attr in self.common_attributes])
            self._matrix = matrix
        return self._matrix


class Modeler(DessiaObject):
    """
    Object that encapsulate standard processes in machine learning modelings.

    `Modeler` object allows to:
        * fit a model from models
        * scale input and output data before fit or predict
        * score a model from models
        * validate a modeling process with cross_validation method
        * plot performances and predictions of a model stored in `Modeler`
        * store a fitted model and associated fitted scaler in a `Modeler` element that can be re-used in another
        workflow as an already trained machine learning model

    :param model:
        Fitted model to make predictions.

    :param input_scaler:
        Scaler for input data.

    :param output_scaler:
        Scaler for output data.

    :param name:
        Name of `Modeler`.
    """

    _standalone_in_db = True
    _allowed_methods = DessiaObject._allowed_methods + ["fit_matrix", "fit_dataset", "predict_matrix",
                                                        "predict_dataset", "fit_predict_matrix", "fit_predict_dataset",
                                                        "score_matrix", "score_dataset", "fit_score_matrix",
                                                        "fit_score_dataset"]

    def __init__(self, model: models.Model, input_scaler: models.Scaler, output_scaler: models.Scaler, name: str = ''):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.in_scaled = self._is_scaled(self.input_scaler)
        self.out_scaled = self._is_scaled(self.output_scaler)
        DessiaObject.__init__(self, name=name)

    def _is_scaled(self, scaler: models.Scaler):
        return not isinstance(scaler, models.IdentityScaler)

    def _format_output(self, scaled_outputs: Matrix):
        """ Format output to list[list[float]] in any case for code consistency and simplicity. """
        if not isinstance(scaled_outputs[0], (list, tuple)):
            return [[value] for value in scaled_outputs]
        return scaled_outputs

    @staticmethod
    def _compute_scalers(inputs: Matrix, outputs: Matrix, input_is_scaled: bool = True, output_is_scaled: bool = False,
                         name: str = '') -> tuple[models.Scaler, models.Scaler, Matrix, Matrix]:
        in_scaler_class, input_scaler_name = models.Scaler.set_in_modeler(name, "in", input_is_scaled)
        out_scaler_class, output_scaler_name = models.Scaler.set_in_modeler(name, "out", output_is_scaled)

        in_scaler, scaled_inputs = in_scaler_class.fit_transform(inputs, input_scaler_name)
        out_scaler, scaled_outputs = out_scaler_class.fit_transform(outputs, output_scaler_name)
        return in_scaler, out_scaler, scaled_inputs, scaled_outputs

    @classmethod
    def _fit(cls, inputs: Matrix, outputs: Matrix, model: models.Model, input_is_scaled: bool = True,
             output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """ Private method to fit outputs to inputs with a machine learning method from datatools.models objects. """
        in_scaler, out_scaler, scaled_inputs, scaled_outputs = cls._compute_scalers(inputs, outputs, input_is_scaled,
                                                                                    output_is_scaled, name)
        fit_model = model.fit(scaled_inputs, scaled_outputs, **model.parameters, name=name + '_model')
        return cls(fit_model, in_scaler, out_scaler, name)

    @classmethod
    def fit_matrix(cls, inputs: Matrix, outputs: Matrix, model: models.Model, input_is_scaled: bool = True,
                   output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Method to fit outputs to inputs with a machine learning method from datatools.models objects for matrix data.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`

        :param input_is_scaled:
            Whether to standardize inputs or not with a `models.StandardScaler`

        :param output_is_scaled:
            Whether to standardize outputs or not with a `models.StandardScaler`

        :param name:
            Name of Modeler

        :return: The equivalent Modeler object containing the fitted model and scalers associated to inputs and outputs
        """
        return cls._fit(inputs, outputs, model, input_is_scaled, output_is_scaled, name)

    @classmethod
    def fit_dataset(cls, dataset: Dataset, input_names: list[str], output_names: list[str], model: models.Model,
                    input_is_scaled: bool = True, output_is_scaled: bool = False, name: str = '') -> 'Modeler':
        """
        Method to fit outputs to inputs with a machine learning method from datatools.models objects for a Dataset.

        :param dataset:
            Dataset containing data, both inputs and outputs

        :param input_names:
            Names of input features

        :param output_names:
            Names of output features

        :param input_is_scaled:
            Whether to standardize inputs or not with a `models.StandardScaler`

        :param output_is_scaled:
            Whether to standardize outputs or not with a `models.StandardScaler`

        :param name:
            Name of Modeler

        :return: The equivalent Modeler object containing the fitted model and scalers associated to inputs and outputs
        """
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return cls.fit_matrix(inputs, outputs, model, input_is_scaled, output_is_scaled, name)

    def _predict(self, inputs: list[list[float]]) -> Union[Vector, Matrix]:
        """ Private method to predict outputs from inputs with self.model. """
        return self.output_scaler.inverse_transform(self.model.predict(self.input_scaler.transform(inputs)))

    def predict_matrix(self, inputs: list[list[float]]) -> Matrix:
        """
        Method to predict outputs from inputs with the current Modeler for matrix data.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`

        :return: The predicted values for inputs.
        """
        return self._format_output(self._predict(inputs))

    def predict_dataset(self, dataset: Dataset, input_names: list[str], output_names: list[str]) -> SampleDataset:
        """
        Method to predict outputs from inputs with the current Modeler for Dataset object.

        :param dataset:
            Dataset containing data, both inputs and outputs

        :param input_names:
            Names of input features to predict

        :param output_names:
            Names of predicted features

        :return: The predicted values for inputs.
        """
        inputs = dataset.sub_matrix(input_names)
        outputs = self.predict_matrix(inputs)
        return SampleDataset.from_matrices(inputs, outputs, input_names, output_names, f'{self.name}_preds')

    @classmethod
    def _fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, model: models.Model,
                     input_is_scaled: bool = True, output_is_scaled: bool = False,
                     name: str = '') -> tuple['Modeler', Union[Vector, Matrix]]:
        """ Private method to fit outputs to inputs and predict `predicted_inputs` for a Dataset. """
        modeler = cls._fit(inputs, outputs, model, input_is_scaled, output_is_scaled, name)
        return modeler, modeler._predict(predicted_inputs)

    @classmethod
    def fit_predict_matrix(cls, inputs: Matrix, outputs: Matrix, predicted_inputs: Matrix, model: models.Model,
                           input_is_scaled: bool = True, output_is_scaled: bool = False,
                           name: str = '') -> tuple['Modeler', Matrix]:
        """ Fit outputs to inputs and predict `predicted_inputs` for matrix data (fit then predict). """
        modeler, predictions = cls._fit_predict(inputs, outputs, predicted_inputs, model, input_is_scaled,
                                                output_is_scaled, name)
        return modeler, modeler._format_output(predictions)

    @classmethod
    def fit_predict_dataset(cls, fit_dataset: Dataset, to_predict_dataset: Dataset, input_names: list[str],
                            output_names: list[str], model: models.Model, input_is_scaled: bool = True,
                            output_is_scaled: bool = False, name: str = '') -> tuple['Modeler', Matrix]:
        """ Fit outputs to inputs and predict outputs of `to_predict_dataset` (fit then predict). """
        modeler = cls.fit_dataset(fit_dataset, input_names, output_names, model, input_is_scaled, output_is_scaled,
                                  name)
        return modeler, modeler.predict_dataset(to_predict_dataset, input_names, output_names)

    def _score(self, inputs: Matrix, outputs: Matrix) -> float:
        """ Compute the score of Modeler. """
        return self.model.score(self.input_scaler.transform(inputs), self.output_scaler.transform(outputs))

    def score_matrix(self, inputs: Matrix, outputs: Matrix) -> float:
        """
        Compute the score of Modeler from matrix.

        Please be sure to fit the model before computing its score and use test data and not train data.
        Train data is data used to train the model and shall not be used to evaluate its quality.
        Test data is data used to test the model and must not be used to train (fit) it.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`

        :return: The score of Modeler.
        """
        return self._score(inputs, outputs)

    def score_dataset(self, dataset: Dataset, input_names: list[str], output_names: list[str]) -> float:
        """
        Compute the score of Modeler from Dataset.

        Please be sure to fit the model before computing its score and use test data and not train data.
        Train data is data used to train the model and shall not be used to evaluate its quality.
        Test data is data used to test the model and must not be used to train (fit) it.

        :param dataset:
            Dataset containing data, both inputs and outputs

        :param input_names:
            Names of input features

        :param output_names:
            Names of output features

        :return: The score of Modeler.
        """
        inputs, outputs = dataset.to_input_output(input_names, output_names)
        return self._score(inputs, outputs)

    @classmethod
    def _fit_score(cls, inputs_train: Matrix, inputs_test: Matrix, outputs_train: Matrix, outputs_test: Matrix,
                   model: models.Model, input_is_scaled: bool, output_is_scaled: bool,
                   name: str) -> tuple['Modeler', float]:
        """ Private method to fit modeler with train matrices and test it with test matrices. """
        mdlr = cls._fit(inputs_train, outputs_train, model, input_is_scaled, output_is_scaled, name)
        return mdlr, mdlr._score(inputs_test, outputs_test)

    @classmethod
    def fit_score_matrix(cls, inputs: Matrix, outputs: Matrix, model: models.Model, input_is_scaled: bool,
                         output_is_scaled: bool, ratio: float = 0.8, name: str = '') -> tuple['Modeler', float]:
        """ Fit modeler with train matrices and test it with test matrices. """
        in_train, in_test, out_train, out_test = models.train_test_split(inputs, outputs, ratio=ratio)
        return cls._fit_score(in_train, in_test, out_train, out_test, model, input_is_scaled,
                              output_is_scaled, name)

    @classmethod
    def fit_score_dataset(cls, dataset: Dataset, input_names: list[str], output_names: list[str], model: models.Model,
                          input_is_scaled: bool = True, output_is_scaled: bool = False, ratio: float = 0.8,
                          name: str = '') -> tuple['Modeler', float]:
        """ Train test split dataset, fit modeler with train matrices and score it with test matrices. """
        train_dataset, test_dataset = dataset.train_test_split(ratio=ratio, shuffled=True)
        inputs_train, output_train = train_dataset.to_input_output(input_names, output_names)
        inputs_test, output_test = test_dataset.to_input_output(input_names, output_names)
        return cls._fit_score(inputs_train, inputs_test, output_train, output_test, model, input_is_scaled,
                              output_is_scaled, name)

class ValidationData(DessiaObject):
    """
    Object that stores modeling data as inputs, outputs and predictions matrices.

    :param inputs:
        Matrix of input data.

    :param outputs:
        Matrix of output data.

    :param predictions:
        Matrix of predicted inputs with a `Modeler`.

    :param input_names:
        Names of input features.

    :param output_names:
        Names of output features.

    :param name:
        Name of `ValidationData`.
    """

    def __init__(self, inputs: Matrix, outputs: Matrix, predictions: Matrix, name: str = ''):
        self.inputs = inputs
        self.outputs = outputs
        self.predictions = predictions
        DessiaObject.__init__(self, name=name)

    def points(self, input_names: list[str], output_names: list[str], reference_path: str) -> Points:
        """ Get output vs prediction for each row of outputs matrix. """
        samples_list = []
        for row, (input_, ref_out, pred_out) in enumerate(zip(self.inputs, self.outputs, self.predictions)):
            values = {attr: input_[col] for col, attr in enumerate(input_names)}
            values.update({f"{attr}_ref": ref_out[col] for col, attr in enumerate(output_names)})
            values.update({f"{attr}_pred": pred_out[col] for col, attr in enumerate(output_names)})
            full_reference_path = f"{reference_path}/dessia_objects/{row}"
            name = f"Sample_{row}"
            samples_list.append(Sample(values=values, reference_path=full_reference_path, name=name))
        return samples_list


class TrainTestData(DessiaObject):
    """
    Object that train and test data to validate modelers.

    :param training_valdata:
        `ValidationData` of training data.

    :param testing_valdata:
        `ValidationData` of testing data.

    :param input_names:
        Names of input features.

    :param output_names:
        Names of output features.

    :param name:
        Name of `TrainTestData`.
    """

    def __init__(self, training_valdata: ValidationData, testing_valdata: ValidationData, input_names: list[str],
                 output_names: list[str], name: str = ''):
        self.training_valdata = training_valdata
        self.testing_valdata = testing_valdata
        self.input_names = input_names
        self.output_names = output_names
        DessiaObject.__init__(self, name=name)

    def _concatenate_outputs(self) -> Matrix:
        return self.training_valdata.outputs + self.testing_valdata.outputs + \
            self.training_valdata.predictions + self.testing_valdata.predictions

    def _matrix_ranges(self) -> Matrix:
        return matrix_ranges(self._concatenate_outputs(), nb_points=10)

    def _ref_pred_names(self) -> list[list[str]]:
        return [[name + '_ref', name + '_pred'] for name in self.output_names]

    def _tooltip(self) -> Tooltip:
        return Tooltip(self.input_names + sum(self._ref_pred_names(), []))

    def _ref_pred_datasets(self, points_train: Points, points_test: Points) -> list[Point2D]:
        ref_args = {'point_style': REF_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Train data'}
        pred_args = {'point_style': VAL_POINT_STYLE, 'edge_style': NO_LINE, 'name': 'Test data'}

        points = [Point2D(sample.values['n_di_ref'], sample.values['n_di_pred'], ref_args['point_style'],
                          reference_path=sample.reference_path,
                          tooltip=f"ref: {sample.values['n_di_ref']} ; pred: {sample.values['n_di_pred']}")
                  for sample in points_train]
        points += [Point2D(sample.values['n_di_ref'], sample.values['n_di_pred'], pred_args['point_style'],
                           reference_path=sample.reference_path,
                           tooltip=f"ref: {sample.values['n_di_ref']} ; pred: {sample.values['n_di_pred']}")
                   for sample in points_test]
        return points

    def _bisectrice_points(self) -> Points:
        hack_bisectrices = []
        for point in zip(*self._matrix_ranges()):
            hack_bisectrices.append({f"{self.output_names[0]}_ref": point[0], f"{self.output_names[0]}_pred": point[0]})
            for idx, name in enumerate(self.output_names):
                hack_bisectrices[-1].update({name + '_ref': point[idx], name + '_pred': point[idx]})
        return hack_bisectrices

    def _to_val_points(self, reference_path: str) -> list[pl_Dataset]:
        return self.training_valdata.points(self.input_names, self.output_names, reference_path), \
            self.testing_valdata.points(self.input_names, self.output_names, reference_path), self._bisectrice_points()

    def build_labels(self) -> list[Label]:
        return [
            Label(title="Train Data", shape=Point2D(0, 0, point_style=REF_POINT_STYLE)),
            Label(title="Test Data", shape=Point2D(0, 0, point_style=VAL_POINT_STYLE)),
            Label(title="y = x", shape=LineSegment2D([0, 0], [1, 1], edge_style=STD_LINE))
        ]

    def build_graphs(self, reference_path: str) -> tuple[PrimitiveGroup, list[Point2D]]:
        """ Build elements and graphs for `plot_data` method. """
        points_train, points_test, points_bisectrice = self._to_val_points(reference_path)
        primitives = [LineSegment2D([points_bisectrice[0]['n_di_ref'], points_bisectrice[0]['n_di_pred']],
                                    [points_bisectrice[-1]['n_di_ref'], points_bisectrice[-1]['n_di_pred']],
                                    edge_style=STD_LINE)]
        primitives += [Point2D(point['n_di_ref'], point['n_di_pred'], point_style=LIN_POINT_STYLE)
                       for point in points_bisectrice]
        primitives += self._ref_pred_datasets(points_train, points_test)
        primitives += self.build_labels()
        return [PrimitiveGroup(primitives, axis_on=True)], points_train + points_test + points_bisectrice

    def plot_data(self, reference_path: str = '#', **_):
        """ Plot data method for `TrainTestData`. """
        graphs, elements = self.build_graphs(reference_path)
        if len(graphs) == 1:
            return graphs
        return [MultiplePlots(elements=elements, plots=graphs, initial_view_on=True)]


class ModelValidation(DessiaObject):
    """ Class to handle a modeler and the `TrainTestData` used to train and test it. """

    _non_data_eq_attributes = ['_score']
    _standalone_in_db = True
    _allowed_methods = DessiaObject._allowed_methods + ["from_matrix", "from_dataset", "scores"]

    def __init__(self, data: TrainTestData, score: float, name: str = ''):
        self.data = data
        self.score = score
        DessiaObject.__init__(self, name=name)
# TODO: is this too heavy ? To merge with TrainTestData ?

    @classmethod
    def _build(cls, modeler: Modeler, input_train: Matrix, input_test: Matrix, output_train: Matrix,
               output_test: Matrix, input_names: list[str], output_names: list[str],
               name: str = '') -> 'ModelValidation':
        trained_mdlr, pred_test = Modeler.fit_predict_matrix(input_train, output_train, input_test, modeler.model,
                                                             modeler.in_scaled, modeler.out_scaled, name)
        pred_train = trained_mdlr.predict_matrix(input_train)
        train_test_data = TrainTestData(ValidationData(input_train, output_train, pred_train),
                                        ValidationData(input_test, output_test, pred_test),
                                        input_names, output_names, f"{name}_data")
        return cls(train_test_data, trained_mdlr.score_matrix(input_test, output_test), name)


    @classmethod
    def from_matrix(cls, modeler: Modeler, inputs: Matrix, outputs: Matrix, input_names: list[str],
                    output_names: list[str], ratio: float = 0.8, name: str = '') -> 'ModelValidation':
        """
        Create a `ModelValidation` object from inputs and outputs matrices.

        :param modeler:
            Modeler type and its hyperparameters, stored in a `Modeler` object for the sake of simplicity. Here,
             modeler does not need to be fitted.

        :param inputs:
            Matrix of data of dimension `n_samples x n_features`

        :param outputs:
            Matrix of data of dimension `n_samples x n_features`

        :param input_names:
            Names of input features

        :param output_names:
            Names of output features

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, `in_train` will be of length `int(ratio)` and `in_test` of
            length `len_matrix - int(ratio)`.

        :param name:
            Name of `ModelValidation`

        :return: A `ModelValidation` object, containing the fitted modeler, its score, train and test data and their
         predictions for input, stored in a `TrainTestData` object.
        """
        in_train, in_test, out_train, out_test = models.train_test_split(inputs, outputs, ratio=ratio)
        return cls._build(modeler, in_train, in_test, out_train, out_test, input_names, output_names, name)

    @classmethod
    def from_dataset(cls, modeler: Modeler, dataset: Dataset, input_names: list[str], output_names: list[str],
                     ratio: float = 0.8, name: str = '') -> 'ModelValidation':
        """
        Create a `ModelValidation` object from a dataset.

        :param modeler:
            Modeler type and its hyperparameters, stored in a `Modeler` object for the sake of simplicity. Here,
             modeler does not need to be fitted.

        :param dataset:
            Dataset containing data, both inputs and outputs

        :param input_names:
            Names of input features

        :param output_names:
            Names of output features

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, `in_train` will be of length `int(ratio)` and `in_test` of
            length `len_matrix - int(ratio)`.

        :param name:
            Name of `ModelValidation`

        :return: A `ModelValidation` object, containing the fitted modeler, its score, train and test data and their
         predictions for input, stored in a TrainTestData object.
        """
        train_dataset, test_dataset = dataset.train_test_split(ratio=ratio, shuffled=True)
        in_train, out_train = train_dataset.to_input_output(input_names, output_names)
        in_test, out_test = test_dataset.to_input_output(input_names, output_names)
        return cls._build(modeler, in_train, in_test, out_train, out_test, input_names, output_names, name)

    def plot_data(self, reference_path: str = '#', **_):
        """ Plot data method for `ModelValidation`. """
        return self.data.plot_data(reference_path=reference_path)


class CrossValidation(DessiaObject):
    """
    Class to cross validate a `Modeler` modeling.

    The purpose of cross validation is to validate a modeling process for a specific type of machine learning
    method, set with specific hyperparameters.
    The first step of cross validation is to split data into train and test data. Then the model is fitted with
    train data and scored with test data. Furthermore, train and test inputs are predicted with the model and
    plotted in a graph that plots these predictions versus reference values. In this plot, the more red points are
    near the black line, the more the model can predict new data precisely.
    This process of cross validation is ran `nb_tests` times. If all of them show a good score and a nice train test
    plot, then the tested modeling is validated and can be used in other, but similar, processes for
    predictions.
    """

    _non_data_eq_attributes = ['_scores']
    _standalone_in_db = True
    _allowed_methods = DessiaObject._allowed_methods + ["from_matrix", "from_dataset", "scores"]

    def __init__(self, model_validations: list[ModelValidation], name: str = ''):
        self.model_validations = model_validations
        self._scores = None
        DessiaObject.__init__(self, name=name)

    @property
    def scores(self) -> Vector:
        """ List of scores of modelers contained in `model_validations`. """
        if self._scores is None:
            self._scores = [model_val.score for model_val in self.model_validations]
        return self._scores

    def _points_scores(self, reference_path: str) -> Points:
        scores = self.scores
        samples_scores = []
        for idx, score in enumerate(scores):
            values = {'Index': idx, 'Score': score}
            full_reference_path = f"{reference_path}/model_validations/{idx}"
            name = f"model_validation_{idx}"
            samples_scores.append(Sample(values=values, reference_path=full_reference_path, name=name))
        return samples_scores

    def _plot_score(self, reference_path: str) -> Graph2D:
        scores = self._points_scores(reference_path)
        nidx = len(scores)
        limits = pl_Dataset(elements=scores_limits(nidx), point_style=INV_POINT_STYLE, edge_style=NO_LINE)
        axis = axis_style(nidx, nidx)

        scores_ds = pl_Dataset(elements=scores, tooltip=Tooltip(['Index', 'Score']), point_style=REF_POINT_STYLE,
                               edge_style=STD_LINE, name="Scores")

        return Graph2D(x_variable='Index', y_variable='Score', graphs=[scores_ds, limits], axis=axis)

    @classmethod
    def from_matrix(cls, modeler: Modeler, inputs: Matrix, outputs: Matrix, input_names: list[str],
                    output_names: list[str], nb_tests: int = 5, ratio: float = 0.8,
                    name: str = '') -> 'CrossValidation':
        """ Cross validation of modeler from inputs and outputs matrices, given `input_names` and `output_names`. """
        validations = [ModelValidation.from_matrix(modeler, inputs, outputs, input_names, output_names, ratio,
                                                   f"{name}_val_{idx}") for idx in range(nb_tests)]
        return cls(validations, name)

    @classmethod
    def from_dataset(cls, modeler: Modeler, dataset: Dataset, input_names: list[str], output_names: list[str],
                     nb_tests: int = 5, ratio: float = 0.8) -> 'CrossValidation':
        """
        Cross validation of modeler from a Dataset object, given `input_names` and `output_names`.

        :param modeler:
            Modeler type and its hyperparameters, stored in a `Modeler` object for the sake of simplicity. Here,
             modeler does not need to be fitted.

        :param dataset:
            Dataset containing data, both inputs and outputs

        :param input_names:
            Names of input features

        :param output_names:
            Names of output features

        :param nb_tests:
            Number of train test validation to run in cross_validation method

        :param ratio:
            Ratio on which to split matrix. If ratio > 1, `in_train` will be of length `int(ratio)` and `in_test` of
            length `len_matrix - int(ratio)`.
        """
        validations = []
        for idx in range(nb_tests):
            name = f"{modeler.name}_val_{idx}"
            validations.append(ModelValidation.from_dataset(modeler, dataset, input_names, output_names, ratio, name))
        return cls(validations, f"{name}_crossval")

    def plot_data(self, reference_path: str = '#', **_):
        """ Plot data method for `CrossValidation`. """
        graphs = []
        for idx, validation in enumerate(self.model_validations):
            graphs += validation.data.build_graphs(reference_path=f"{reference_path}/model_validations/{idx}")[0]
        return [self._plot_score(reference_path=reference_path),
                MultiplePlots(graphs, elements=[{"factice_key":0}], initial_view_on=True)]


def matrix_ranges(matrix: Matrix, nb_points: int = 20) -> Matrix:
    """ Dessia linspace of `nb_points` points between extrema of each column of matrix. """
    ranges = []
    for feature_column in zip(*matrix):
        min_value = min(feature_column)
        max_value = max(feature_column)
        step_range = (max_value - min_value)/nb_points
        ranges.append(npy.arange(min_value, max_value, step_range).tolist() + [1.05 * max_value])
    return ranges

def axis_style(nb_x: int = 10, nb_y: int = 10) -> Axis:
    """ Set axis style for `Modeler` objects. """
    return Axis(nb_points_x=nb_x, nb_points_y=nb_y, axis_style=STD_LINE, grid_on=True)

def scores_limits(number: int) -> Points:
    """ Draw white points in scatter for it to be plotted between 0 and number on x axis and 0 and 1 on y axis. """
    return [{'Index': -0.05, 'Score': -0.05}, {'Index': number + 0.05, 'Score': 1.05}]
