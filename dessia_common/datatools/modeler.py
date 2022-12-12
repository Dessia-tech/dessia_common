"""
Librairy for building machine learning models from Dataset or Lists using sklearn models handled in models.
"""
from typing import List, Dict, Any, Tuple, Union, Type

import numpy as npy

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
            return models.StandardScaler()
        return models.IdentityScaler()

    @staticmethod
    def _set_scaler_name(modeler_name: str, in_out: str, is_scaled: bool) -> str:
        name = f"{modeler_name}_"
        return name + (f"{in_out}_scaler" if is_scaled else "indentity_scaler")

    @staticmethod
    def _set_scaler(modeler_name: str, in_out: str, is_scaled: bool) -> models.Scaler:
        class_ = Modeler._set_scaler_class(is_scaled)
        name = Modeler._set_scaler_name(modeler_name, in_out, is_scaled)
        return class_, name

    @staticmethod
    def _dataset_to_x_y(dataset: Dataset, input_names:List[str], output_names: List[str]):
        return

    @classmethod
    def fit_matrix(cls, inputs: Matrix, outputs: Matrix, class_: Type, hyperparameters: Dict[str, Any],
            input_is_scaled: bool = True, output_is_scaled: bool = False, name: str = '') -> 'Modeler':

        in_scaler_class, input_scaler_name = cls._set_scaler(name, "in", input_is_scaled)
        out_scaler_class, output_scaler_name = cls._set_scaler(name, "out", output_is_scaled)

        in_scaler, scaled_inputs = in_scaler_class.fit_transform(inputs, input_scaler_name)
        out_scaler, scaled_outputs = out_scaler_class.fit_transform(outputs, output_scaler_name)

        model = class_.fit(scaled_inputs, scaled_outputs, **hyperparameters, name=name + '_model')
        return cls(model=model, input_scaler=in_scaler, output_scaler=out_scaler, name=name)

    def predict(self, inputs: List[List[float]]) -> Matrix:
        scaled_inputs = self.input_scaler.transform(inputs)
        scaled_outputs = self.model.predict(scaled_inputs)
        return self.output_scaler.inverse_transform(scaled_outputs)

    @classmethod
    def fit_predict(cls, inputs: Matrix, outputs: Matrix, predicted_outputs: Matrix, class_: Type,
                    hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                    name: str = '') -> Tuple['Modeler', Matrix]:

        modeler = cls.fit(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)
        return modeler, modeler.predict(predicted_outputs)

    @classmethod
    def fit_dataset(cls, dataset: Dataset, input_names: List[str], output_names: List[str], class_: Type,
                         hyperparameters: Dict[str, Any], input_is_scaled: bool = True, output_is_scaled: bool = False,
                         name: str = '') -> 'Modeler':
        inputs = dataset.sub_matrix(input_names)
        outputs = dataset.sub_matrix(output_names)
        return cls.fit(inputs, outputs, class_, hyperparameters, input_is_scaled, output_is_scaled, name)

    def predict_dataset(self, dataset: Dataset, input_names: List[str]) -> Matrix:
        inputs = dataset.sub_matrix(input_names)
        return self.predict(inputs)

    @classmethod
    def fit_predict_dataset(cls, dataset: Dataset, input_names: List[str], output_names: List[str],
                            predicted_names: List[str], class_: Type, hyperparameters: Dict[str, Any],
                            input_is_scaled: bool = True, output_is_scaled: bool = False,
                            name: str = '') -> Tuple['Modeler', Matrix]:

        modeler = cls.fit(dataset, input_names, output_names, class_, hyperparameters, input_is_scaled,
                          output_is_scaled, name)
        return modeler, modeler.predict(predicted_names)




