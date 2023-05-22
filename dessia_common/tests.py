#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Test module for dessia_common. """
from time import sleep
from typing import List, Tuple
import random
import numpy as npy
import matplotlib.pyplot as plt
from dessia_common.core import DessiaObject
import dessia_common.measures as dcm
import dessia_common.files as dcf


class Submodel(DessiaObject):
    """ Mock a MBSE that is a standalone attribute of Model, for testing purpose. """

    _standalone_in_db = True

    def __init__(self, subvalue: int, name: str = ''):
        self.subvalue = subvalue
        self.name = name

        DessiaObject.__init__(self, name=name)


class Model(DessiaObject):
    """ Mock a standalone MBSE for testing purpose. """

    _standalone_in_db = True

    def __init__(self, value: int, submodel: Submodel, name: str = ''):
        self.value = value
        self.submodel = submodel

        DessiaObject.__init__(self, name=name)


class Generator(DessiaObject):
    """ Mock a generator of MBSEs for testing purpose. """

    _standalone_in_db = True
    _allowed_methods = ['long_generation']

    def __init__(self, parameter: int, nb_solutions: int = 25, models: List[Model] = None, name: str = ''):
        self.parameter = parameter
        self.nb_solutions = nb_solutions
        self.models = models

        DessiaObject.__init__(self, name=name)

    def generate(self) -> None:
        """Generate models."""
        submodels = [Submodel(self.parameter * i) for i in range(self.nb_solutions)]
        self.models = [Model(self.parameter + i, submodels[i]) for i in range(self.nb_solutions)]

    def long_generation(self, progress_callback=lambda x: None) -> List[Model]:
        """
        Run a long generation.

        This method aims to test:
            * lots of prints to be caught
            * progress update
            * long computation
        """
        submodels = [Submodel(self.parameter * i) for i in range(self.nb_solutions)]
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
    """ Mock an optimizer for testing purpose. """

    _standalone_in_db = True

    def __init__(self, model_to_optimize: Model, name: str = ''):
        self.model_to_optimize = model_to_optimize

        DessiaObject.__init__(self, name=name)

    def optimize(self, optimization_value: int = 3) -> None:
        """Optimize attribute model_to_optimize."""
        self.model_to_optimize.value += optimization_value


class Component(DessiaObject):
    """ MBSE to be tested. Mock a real usecase. For example, a module in a battery pack. """

    _standalone_in_db = True

    def __init__(self, efficiency, name: str = ''):
        self.efficiency = efficiency
        DessiaObject.__init__(self, name=name)

    def power_simulation(self, power_value: dcm.Power):
        """ Run a power simulation. """
        return power_value * self.efficiency


class ComponentConnection(DessiaObject):
    """ Mock for a link between to components, e.g. the electrical wiring between two modules in a battery pack. """

    def __init__(self, input_component: Component, output_component: Component, name: str = ''):
        self.input_component = input_component
        self.output_component = output_component
        DessiaObject.__init__(self, name=name)


class SystemUsage(DessiaObject):
    """  Mock a simulation result of a system, e.g. the response of the battery pack to a certain use case. """

    _standalone_in_db = True

    def __init__(self, time: List[dcm.Time], power: List[dcm.Power], name: str = ''):
        self.time = time
        self.power = power
        DessiaObject.__init__(self, name=name)


class System(DessiaObject):
    """ Mock a system that binds several components and their connection. For example, a battery pack. """

    _standalone_in_db = True
    _dessia_methods = ['power_simulation']

    def __init__(self, components: List[Component], component_connections: List[ComponentConnection], name: str = ''):
        self.components = components
        self.component_connections = component_connections
        DessiaObject.__init__(self, name=name)

    def output_power(self, input_power: dcm.Power):
        """ Compute output power. """
        return input_power * 0.8

    def power_simulation(self, usage: SystemUsage):
        """ Run a power simulation. """
        output_power = []
        for _, input_power in zip(usage.time, usage.power):
            output_power.append(self.output_power(input_power))
        return SystemSimulationResult(self, usage, output_power)


class SystemSimulationResult(DessiaObject):
    """ Result wrapper for a System with its usage. """

    _standalone_in_db = True

    def __init__(self, system: System, system_usage: SystemUsage, output_power: List[dcm.Power], name: str = ''):
        self.system = system
        self.system_usage = system_usage
        self.output_power = output_power
        DessiaObject.__init__(self, name=name)


class SystemSimulationList(DessiaObject):
    """ Results container for system simulation results. """

    _standalone_in_db = True

    def __init__(self, simulations: List[SystemSimulationResult], name: str = ''):
        self.simulations = simulations
        DessiaObject.__init__(self, name=name)


class Car(DessiaObject):
    """ Defines a car with all its features. """

    _standalone_in_db = True
    _non_data_hash_attributes = ['name']

    def __init__(self, name: str, mpg: float, cylinders: int, displacement: dcm.Distance, horsepower: float,
                 weight: dcm.Mass, acceleration: dcm.Time, model: int, origin: str):
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
        """ Get equivalent vector of instance of Car. """
        list_formated_car = []
        for feature in self.vector_features():
            list_formated_car.append(getattr(self, feature.lower()))
        return list_formated_car

    @classmethod
    def from_csv(cls, file: dcf.StringFile, end: int = None, remove_duplicates: bool = False):
        """ Generates Cars from given .csv file. """
        array = [row.split(',') for row in file.getvalue().split('\n')][1:-1]
        cars = []
        for idx_line, line in enumerate(array):
            if end is not None and idx_line >= end:
                break
            if not remove_duplicates or (remove_duplicates and line.tolist() not in cars):
                attr_list = [float(attr) if attr.replace('.', '').isnumeric() else attr for attr in line]
                attr_list[3] /= 1000

                for idx_attr, attr in enumerate(attr_list):
                    if isinstance(attr, npy.int64):
                        attr_list[idx_attr] = int(attr)
                    elif isinstance(attr, npy.float64):
                        attr_list[idx_attr] = float(attr)
                cars.append(cls(*attr_list))
        return cars


class CarWithFeatures(Car):
    """ Defines a car with all specific features (`'mpg', 'displacement', 'horsepower', 'acceleration', 'weight'`). """

    _vector_features = ['mpg', 'displacement', 'horsepower', 'acceleration', 'weight']

    def __init__(self, name: str, mpg: float, cylinders: int, displacement: dcm.Distance, horsepower: float,
                 weight: dcm.Mass, acceleration: dcm.Time, model: int, origin: str):
        Car.__init__(self, name, mpg, cylinders, displacement, horsepower, weight, acceleration, model, origin)

    @classmethod
    def vector_features(cls):
        """ Get list of _vector_features. """
        return cls._vector_features


class RandDataD1(DessiaObject):
    """ Creates a dataset with 1 parameters from a number of clusters and dimensions. """

    _standalone_in_db = True
    _non_data_hash_attributes = ['name']
    _nb_dims = 1
    _vector_features = [f'p_{i+1}' for i in range(_nb_dims)]

    def __init__(self, p_1: float, name: str = ''):
        DessiaObject.__init__(self, name=name)
        self.p_1 = p_1

    @classmethod
    def vector_features(cls):
        """Get list of _vector_features."""
        return cls._vector_features

    @classmethod
    def create_dataset(cls, nb_clusters: int = 10, nb_points: int = 2500, mean_borns: Tuple[float, float] = (-50., 50),
                       std_borns: Tuple[float, float] = (-2., 2.)):
        """ Create a random dataset with a number of clusters, number of points, means and std per cluster. """
        means_list = []
        std_list = []
        data_list = []
        cluster_sizes = cls.set_cluster_sizes(nb_points, nb_clusters)

        for cluster_size in cluster_sizes:
            means_list = [random.uniform(*mean_borns) for i in range(cls._nb_dims)]
            std_list = [random.uniform(*std_borns) for i in range(cls._nb_dims)]
            for _ in range(cluster_size):
                data_list.append(
                    cls(*[random.normalvariate(means_list[dim], std_list[dim]) for dim in range(cls._nb_dims)]))

        return data_list

    @staticmethod
    def set_cluster_sizes(nb_points: int, nb_clusters: int):
        """ Set clusters' sizes in function of total number of points. """
        current_nb_points = nb_points
        cluster_sizes = []
        for _ in range(nb_clusters - 1):
            points_in_cluster = random.randint(int(current_nb_points / nb_clusters / 2),
                                               int(current_nb_points / nb_clusters * 2))
            cluster_sizes.append(points_in_cluster)
            current_nb_points -= points_in_cluster

        cluster_sizes.append(int(nb_points - npy.sum(cluster_sizes)))
        return cluster_sizes

    @staticmethod
    def python_plot(x_label: str, y_label: str, rand_data_list: List[List[float]], **kwargs):
        """Plot with matplotlib."""
        x_coords = [getattr(RandData, x_label) for RandData in rand_data_list]
        y_coords = [getattr(RandData, y_label) for RandData in rand_data_list]
        plt.plot(x_coords, y_coords, **kwargs)


class RandDataD2(RandDataD1):
    """ Creates a dataset with 2 parameters from a number of clusters and dimensions. """

    _nb_dims = 2
    _vector_features = [f'p_{i+1}' for i in range(2)]

    def __init__(self, p_1: float, p_2: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2


class RandDataD3(RandDataD1):
    """ Creates a dataset with 3 parameters from a number of clusters and dimensions. """

    _nb_dims = 3
    _vector_features = [f'p_{i+1}' for i in range(3)]

    def __init__(self, p_1: float, p_2: float, p_3: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3


class RandDataD4(RandDataD1):
    """ Creates a dataset with 4 parameters from a number of clusters and dimensions. """

    _nb_dims = 4
    _vector_features = [f'p_{i+1}' for i in range(4)] + ['test_prop']

    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self._test_prop = None

    @property
    def test_prop(self):
        """Factice property for some tests."""
        if self._test_prop is None:
            self._test_prop = 3
        return self._test_prop


class RandDataD5(RandDataD1):
    """ Creates a dataset with 5 parameters from a number of clusters and dimensions. """

    _nb_dims = 5
    _vector_features = [f'p_{i+1}' for i in range(5)] + ['test_prop']

    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, p_5: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self._test_prop = None

    @property
    def test_prop(self):
        """Factice property for some tests."""
        if self._test_prop is None:
            self._test_prop = 3
        return self._test_prop


class RandDataD6(RandDataD1):
    """ Creates a dataset with 6 parameters from a number of clusters and dimensions. """

    _nb_dims = 6
    _vector_features = [f'p_{i+1}' for i in range(6)] + ['test_prop']

    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, p_5: float, p_6: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self.p_6 = p_6
        self._test_prop = None

    @property
    def test_prop(self):
        """ Factice property for some tests. """
        if self._test_prop is None:
            self._test_prop = 3
        return self._test_prop


class RandDataD7(RandDataD1):
    """ Creates a dataset with 7 parameters from a number of clusters and dimensions. """

    _nb_dims = 7
    _vector_features = [f'p_{i+1}' for i in range(7)]
    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, p_5: float, p_6: float, p_7: float,
                 name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self.p_6 = p_6
        self.p_7 = p_7


class RandDataD8(RandDataD1):
    """ Creates a dataset with 8 parameters from a number of clusters and dimensions. """

    _nb_dims = 8
    _vector_features = [f'p_{i+1}' for i in range(8)]
    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, p_5: float, p_6: float, p_7: float, p_8: float,
                 name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self.p_6 = p_6
        self.p_7 = p_7
        self.p_8 = p_8


class RandDataD9(RandDataD1):
    """ Creates a dataset with 9 parameters from a number of clusters and dimensions. """

    _nb_dims = 9
    _vector_features = [f'p_{i+1}' for i in range(9)]
    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, p_5: float, p_6: float, p_7: float, p_8: float,
                 p_9: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self.p_6 = p_6
        self.p_7 = p_7
        self.p_8 = p_8
        self.p_9 = p_9


class RandDataD10(RandDataD1):
    """ Creates a dataset with 10 parameters from a number of clusters and dimensions. """

    _nb_dims = 10
    _vector_features = [f'p_{i+1}' for i in range(10)]

    def __init__(self, p_1: float, p_2: float, p_3: float, p_4: float, p_5: float, p_6: float, p_7: float, p_8: float,
                 p_9: float, p_10: float, name: str = ''):
        RandDataD1.__init__(self, p_1, name=name)
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self.p_6 = p_6
        self.p_7 = p_7
        self.p_8 = p_8
        self.p_9 = p_9
        self.p_10 = p_10
