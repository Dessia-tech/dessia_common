#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for dessia_common

"""
from time import sleep
from typing import List
import random
import numpy as npy
from dessia_common import DessiaObject
import dessia_common.typings as dct
import dessia_common.files as dcf


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

    def __init__(self, parameter: int, nb_solutions: int = 25, name: str = ''):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        self.models = None

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
        models = [Model(self.parameter + i, submodels[i])
                  for i in range(self.nb_solutions)]
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
    _export_features = ['mpg', 'displacement',
                        'horsepower', 'acceleration', 'weight']
    _non_data_hash_attributes = ['name']

    def __init__(self, name: str, mpg: float, cylinders: int,
                 displacement: dct.Distance, horsepower: float,
                 weight: dct.Mass, acceleration: dct.Time, model: int,
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
    def from_csv(cls, file: dcf.StringFile, end: int = None, remove_duplicates: bool = False):
        """
        Generates Cars from given .csv file.
        """
        array = npy.genfromtxt(
            file, dtype=None, delimiter=',', names=True, encoding=None)
        cars = []
        for i, line in enumerate(array):
            if end is not None and i >= end:
                break
            if not remove_duplicates or (remove_duplicates and line.tolist() not in cars):
                attr_list = list(line)
                attr_list[3] /= 1000

                for i in range(len(attr_list)):
                    if isinstance(attr_list[i], npy.int64):
                        attr_list[i] = int(attr_list[i])
                    elif isinstance(attr_list[i], npy.float64):
                        attr_list[i] = float(attr_list[i])

                cars.append(cls(*attr_list))
        return cars


class Worms(DessiaObject):
    """
    Defines a dataset of worms from file
    """
    _standalone_in_db = True
    _export_features = ['p1', 'p2']
    _non_data_hash_attributes = ['name']

    def __init__(self, name: str, p1: float, p2: float):
        DessiaObject.__init__(self, name=name)
        self.p1 = p1
        self.p2 = p2

    @classmethod
    def from_file(cls, file: dcf.StringFile, delimiter: str = ',', end: int = None, remove_duplicates: bool = False):
        """
        Generates Cars from given .csv file.
        """
        array = npy.genfromtxt(
            file, dtype=None, delimiter=delimiter, names=True, encoding=None)
        dataset = []
        for i, line in enumerate(array):
            if end is not None and i >= end:
                break
            if not remove_duplicates or (remove_duplicates and line.tolist() not in dataset):
                attr_list = list(line)

                for j in range(len(attr_list)):
                    if isinstance(attr_list[j], npy.int64):
                        attr_list[j] = int(attr_list[j])
                    elif isinstance(attr_list[j], npy.float64):
                        attr_list[j] = float(attr_list[j])

                dataset.append(cls(str(i), *attr_list))
        return dataset

    def to_vector(self):
        list_formated_worm = []
        for feature in self._export_features:
            list_formated_worm.append(getattr(self, feature.lower()))
        return list_formated_worm


class ClusTester(DessiaObject):
    """
    Creates a dataset from a number of clusters and dimensions
    """
    _standalone_in_db = True
    _export_features = []
    _non_data_hash_attributes = ['name']

    def __init__(self, p1: float = None, p2: float = None, p3: float = None,
                 p4: float = None, p5: float = None, p6: float = None, p7: float = None,
                 p8: float = None, p9: float = None, p10: float = None, name: str = ''):
        DessiaObject.__init__(self, name=name)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9
        self.p10 = p10

    @classmethod
    def create_dataset(cls, nb_clusters: float = 10, nb_dims: int = 10, nb_points: int = 2500):
        means_list = []
        std_list = []
        data_list = []
        list_attr = [f'p{i+1}' for i in range(nb_dims)]
        cluster_sizes = cls.set_cluster_sizes(nb_points, nb_clusters)

        for cluster_size in cluster_sizes:
            means_list = [random.uniform(-50, 50) for i in range(nb_dims)]
            std_list = [random.uniform(-5, 5) for i in range(nb_dims)]
            for idx_point in range(cluster_size):
                new_data = cls(
                    *[random.normalvariate(means_list[dim], std_list[dim]) for dim in range(nb_dims)])
                new_data.set_export_features(list_attr)
                data_list.append(new_data)

        return data_list

    @staticmethod
    def set_cluster_sizes(nb_points: int, nb_clusters: int):
        current_nb_points = nb_points
        cluster_sizes = []
        for i in range(nb_clusters - 1):
            points_in_cluster = random.randint(int(current_nb_points / nb_clusters / 2),
                                               int(current_nb_points / nb_clusters * 2))
            cluster_sizes.append(points_in_cluster)
            current_nb_points -= points_in_cluster

        cluster_sizes.append(nb_points - npy.sum(cluster_sizes))
        return cluster_sizes

    @staticmethod
    def plot(x_label: str, y_label: str, clustester_list: List[List[float]], **kwargs):
        x_coords = [getattr(clustester, x_label)
                    for clustester in clustester_list]
        y_coords = [getattr(clustester, y_label)
                    for clustester in clustester_list]
        import matplotlib.pyplot as plt
        plt.plot(x_coords, y_coords, **kwargs)
        return

    def set_export_features(self, list_attr: List[str]):
        self._export_features = list_attr

    def to_vector(self):
        clustester_vectored = []
        for feature in self._export_features:
            clustester_vectored.append(getattr(self, feature.lower()))
        return clustester_vectored
