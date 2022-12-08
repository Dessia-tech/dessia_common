"""
Librairy for building machine learning models from Dataset or Lists using sklearn models handled in models.
"""
from typing import List, Dict, Any, Tuple, Union, Type

import numpy as npy

from dessia_common.core import DessiaObject
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
    def _initialize_scaler(is_scaled: bool):
        if is_scaled:
            return models.StandardScaler()
        return models.IdentityScaler()

    @staticmethod
    def _fit_scaler(scaler: models.Scaler, matrix: Matrix, name: str = '') -> 'models.Scaler':
        return scaler.fit(matrix, name=name)

    @staticmethod
    def _transform(scaler: models.Scaler, matrix: Matrix) -> Matrix:
        return scaler.transform(matrix)

    @staticmethod
    def _fit_transform_scaler(scaler: models.Scaler, matrix: Matrix, name: str = '') -> Tuple['models.Scaler', Matrix]:
        return scaler.fit_transform(matrix, name=name)

    # def _initialize_model(self):
    #     return models.Model()

    # def _set_model_attributes(self, model, attributes: Dict[str, float]):
    #     for attr, value in attributes.items():
    #         setattr(model, attr, value)
    #     return model

    # def _instantiate_model(self):
    #     model = self._init_model()
    #     model = self._set_model_attributes(model, self.model_attributes)
    #     model = self._set_model_attributes(model, self.model_)
    #     return model

    @classmethod
    def fit(cls, inputs: List[List[float]], outputs: List[List[float]], class_: Type, hyperparameters: Dict[str, Any],
            input_is_scaled: bool = True, output_is_scaled: bool = False, name: str = '') -> 'Modeler':

        in_scaler, in_scaled = cls._fit_transform_scaler(cls._initialize_scaler(input_is_scaled), inputs,
                                                         (name + '_in_scaler' if output_is_scaled else "idty_scale"))
        out_scaler, out_scaled = cls._fit_transform_scaler(cls._initialize_scaler(output_is_scaled), outputs,
                                                           (name + '_out_scaler' if output_is_scaled else "idty_scale"))

        model = class_.fit(in_scaled, out_scaled, **hyperparameters, name=name + '_model')
        return cls(model=model, input_scaler=in_scaler, output_scaler=out_scaler, name=name)

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
