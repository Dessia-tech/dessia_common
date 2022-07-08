#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for dessia_common

"""
from time import sleep
from typing import List
from io import StringIO
import numpy as npy
from dessia_common import DessiaObject
import dessia_common.typings as dct


class Submodel(DessiaObject):
    _standalone_in_db = True

    def __init__(self, subvalue: int, name: str = ''):
        self.subvalue = subvalue
        self.name = name

        DessiaObject.__init__(self, name=name)


class Model(DessiaObject):
    _standalone_in_db = True

    def __init__(self, value: int, submodel: Submodel, name: str = ''):
        self.value = value
        self.submodel = submodel

        DessiaObject.__init__(self, name=name)


class Generator(DessiaObject):
    _standalone_in_db = True
    _allowed_methods = ['long_generation']

    def __init__(self, parameter: int, nb_solutions: int = 25, models: List[Model] = None, name: str = ''):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        self.models = models

        DessiaObject.__init__(self, name=name)

    def generate(self) -> None:
        submodels = [Submodel(self.parameter * i)
                     for i in range(self.nb_solutions)]
        self.models = [Model(self.parameter + i, submodels[i])
                       for i in range(self.nb_solutions)]

    def long_generation(self, progress_callback=lambda x: None) -> List[Model]:
        """
        This method aims to test:
            * lots of prints to be catched
            * progress update
            * long computation
        """
        submodels = [Submodel(self.parameter * i)
                     for i in range(self.nb_solutions)]
        models = [Model(self.parameter + i, submodels[i]) for i in range(self.nb_solutions)]
        # Delay to simulate long generateion
        print('Beginning a long generation...')
        for i in range(500):
            print(f'Loop n°{i+1} / 500')
            progress = i / 499.
            progress_callback(progress)
            sleep(0.3)
        print('Generation complete')
        return models


class Optimizer(DessiaObject):
    _standalone_in_db = True

    def __init__(self, model_to_optimize: Model, name: str = ''):
        self.model_to_optimize = model_to_optimize

        DessiaObject.__init__(self, name=name)

    def optimize(self, optimization_value: int = 3) -> None:
        self.model_to_optimize.value += optimization_value


class Component(DessiaObject):
    _standalone_in_db = True

    def __init__(self, efficiency, name: str = ''):
        self.efficiency = efficiency
        DessiaObject.__init__(self, name=name)

    def power_simulation(self, power_value: dct.Power):
        return power_value * self.efficiency


class ComponentConnection(DessiaObject):
    def __init__(self, input_component: Component,
                 output_component: Component, name: str = ''):
        self.input_component = input_component
        self.output_component = output_component
        DessiaObject.__init__(self, name=name)


class SystemUsage(DessiaObject):
    _standalone_in_db = True

    def __init__(self, time: List[dct.Time], power: List[dct.Power],
                 name: str = ''):
        self.time = time
        self.power = power
        DessiaObject.__init__(self, name=name)


class System(DessiaObject):
    _standalone_in_db = True
    _dessia_methods = ['power_simulation']

    def __init__(self, components: List[Component],
                 component_connections: List[ComponentConnection],
                 name: str = ''):
        self.components = components
        self.component_connections = component_connections
        DessiaObject.__init__(self, name=name)

    def output_power(self, input_power: dct.Power):
        return input_power * 0.8

    def power_simulation(self, usage: SystemUsage):
        output_power = []
        for _, input_power in zip(usage.time, usage.power):
            output_power.append(self.output_power(input_power))
        return SystemSimulationResult(self, usage, output_power)


class SystemSimulationResult(DessiaObject):
    _standalone_in_db = True

    def __init__(self, system: System, system_usage: SystemUsage,
                 output_power: List[dct.Power], name: str = ''):
        self.system = system
        self.system_usage = system_usage
        self.output_power = output_power
        DessiaObject.__init__(self, name=name)


class SystemSimulationList(DessiaObject):
    _standalone_in_db = True

    def __init__(self, simulations: List[SystemSimulationResult],
                 name: str = ''):
        self.simulations = simulations
        DessiaObject.__init__(self, name=name)


class Car(DessiaObject):
    """
    Defines a car
    """
    _standalone_in_db = True
    _export_features = ['mpg', 'cylinders', 'displacement', 'horsepower',
                        'weight', 'acceleration', 'model']

    def __init__(self, name: str, mpg: float, cylinders: float,
                 displacement: dct.Distance, horsepower: float,
                 weight: dct.Mass, acceleration: dct.Time, model: float,
                 origin: str):
        DessiaObject.__init__(self, name=name)

        self.mpg = mpg
        self.cylinders = cylinders
        self.displacement = displacement
        self.horsepower = horsepower
        self.weight = weight
        self.acceleration = acceleration
        self.model = model
        self.origin = origin

    def to_vector(self):
        list_formated_car = []
        for feature in self._export_features:
            list_formated_car.append(getattr(self, feature.lower()))

        return list_formated_car

    @classmethod
    def from_csv(cls, file: StringIO, end: int = None, remove_duplicates: bool = False):
        """
        Generates Cars from given .csv file.
        """
        array = npy.genfromtxt(file, dtype=None, delimiter=',', names=True, encoding=None)
        variables = list(array.dtype.fields.keys())
        cars = []
        for i, line in enumerate(array):
            if end is not None and i >= end:
                break
            if not remove_duplicates or (remove_duplicates and line.tolist() not in cars):
                attr_list = list(line)
                attr_list[3] /= 1000
                cars.append(cls(*attr_list))

        return cars, variables
