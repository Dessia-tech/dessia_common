#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:56:12 2020

@author: jezequel
"""

import math
from typing import List, Dict
import numpy as np
import pandas as pd
from dessia_common import DessiaObject, Parameter
from scipy.optimize import minimize
from pyDOE import lhs


class ParetoSettings(DessiaObject):
    _generic_eq = True
    _ordered_attributes = ['name', 'enabled', 'minimized_attributes']

    def __init__(self, minimized_attributes: Dict[str, bool], enabled: bool = True, name: str = ''):
        self.enabled = enabled
        self.minimized_attributes = minimized_attributes

        DessiaObject.__init__(self, name=name)


class ObjectiveSettings(DessiaObject):
    _generic_eq = True
    _ordered_attributes = ['name', 'enabled', 'n_near_values']

    def __init__(self, n_near_values: int = 4, enabled: bool = True, name: str = ''):
        self.n_near_values = n_near_values
        self.enabled = enabled

        DessiaObject.__init__(self, name=name)


class Objective(DessiaObject):
    """
    Defines an objective function

    :param coeff_names: List of strings representing ordered variables names.
    :type coeff_names: [str]
    :param coeff_values: List of floats representing coefficients.
    :type coeff_values: [float]

    Order is kept. It means that coefficients are applied to the variables
    in the same order as they are defined.
    """
    _generic_eq = True
    _standalone_in_db = False
    _ordered_attributes = ['name', 'settings', 'coefficients']
    _non_serializable_attributes = ['coeff_names']

    def __init__(self, coefficients: Dict[str, float], settings: ObjectiveSettings, name: str = ''):
        self.coefficients = coefficients
        self.coeff_names = list(coefficients.keys())
        self.settings = settings

        DessiaObject.__init__(self, name=name)

    def apply_individual(self, values):
        rating = 0
        for variable, value in values.items():
            coefficient = self.coefficients[variable]
            rating += coefficient*value
        return rating

    @classmethod
    def coefficients_from_angles(cls, angles: List[float]) -> List[float]:
        """
        compute coefficients from n-1 angles
        """
        n = len(angles) + 1
        matrix = np.identity(n)
        for i, angle in enumerate(angles):
            matrix_i = np.identity(n)
            matrix_i[i, i] = math.cos(angle)
            matrix_i[i+1, i] = -math.sin(angle)
            matrix_i[i, i+1] = math.sin(angle)
            matrix_i[i+1, i+1] = math.cos(angle)
            matrix = np.dot(matrix, matrix_i)
        x = np.zeros(n)
        x[0] = 1
        signed = np.dot(matrix.T, x).tolist()
        unsigned = [abs(v) for v in signed]
        return unsigned

    @classmethod
    def from_angles(cls, angles, variables, settings=None, name="Generated from angles"):
        if not isinstance(angles, list) and not isinstance(angles, np.ndarray):
            angles = [angles]
        generated_coefficients = cls.coefficients_from_angles(angles=angles)
        coefficients = {var: generated_coefficients[i] for i, var in enumerate(variables)}

        if settings is None:
            settings = ObjectiveSettings()

        return Objective(coefficients=coefficients, settings=settings, name=name)


class Catalog(DessiaObject):
    """
    Defines a Catalog object that gathers a collection of VectoredObjects

    :param pareto_attributes: List of strings representing names of variables
                              used for pareto computations.
    :type pareto_attributes: [str]
    :param minimize: List of booleans representing if pareto for this variable
                     should be searched in maximum or minimum direction.
    :type minimize: [bool]
    :param objectives: List of objectives to apply to catalog vectored objects
    :type objectives: [Objective]
    :param n_near_values: Integer that gives the number of best solutions given by objectives
    :type n_near_values: int
    # :param objects: List of vectored objects.
    # :type objects: [VectoredObject]
    :param choice_variables: List of string. List of variable names that represent choice arguments
    :type choice_variables: [str]
    :param name: Name of the catalog
    :type name: str

    """
    _generic_eq = True
    _standalone_in_db = True
    _ordered_attributes = ['name', 'pareto_settings', 'objectives']
    _non_editable_attributes = ['array', 'variables', 'choice_variables']
    _export_formats = ['csv']
    _allowed_methods = ['find_best_objective']
    _whitelist_attributes = ['variables']

    def __init__(self, array: List[List[float]], variables: List[str],
                 pareto_settings: ParetoSettings,
                 objectives: List[Objective],
                 choice_variables: List[str] = None,
                 name: str = ''):
        DessiaObject.__init__(self, name=name)

        self.array = array
        self.variables = variables
        if choice_variables is None:
            self.choice_variables = variables
        else:
            self.choice_variables = choice_variables

        self.pareto_settings = pareto_settings

        self.objectives = objectives
        self.generated_best_objectives = 0

    def _display_angular(self):
        """
        Configures catalog display on frontend

        :return: List of displays dictionnaries
        """
        filters = [{'attribute': variable, 'operator': 'gt', 'bound': 0} for j, variable in enumerate(self.variables)
                   if not isinstance(self.array[0][j], str) and variable in self.choice_variables]

        # Pareto
        costs = self.build_costs(self.pareto_settings)
        pareto_indices = pareto_frontier(costs=costs)

        all_near_indices = {}
        objective_ratings = {}
        for iobjective, objective in enumerate(self.objectives):
            if objective.settings.enabled:
                ratings = self.handle_objective(objective)
                if objective.name and objective.name not in objective_ratings:
                    name = objective.name
                else:
                    name = 'objective_'+str(iobjective)
                objective_ratings[name] = ratings
                filters.append({'attribute': name, 'operator': 'gte', 'bound': 0})
                threshold = objective.settings.n_near_values
                near_indices = list(np.argpartition(ratings, threshold)[:threshold])
                all_near_indices[iobjective] = near_indices

        datasets = []

        values = []
        for i, line in enumerate(self.array):
            value = {}
            for variable in self.variables:
                value[variable] = self.get_value_by_name(line, variable)
            for objective_name, ratings in objective_ratings.items():
                value[objective_name] = ratings[i]
            values.append(value)

        # Pareto
        if self.pareto_settings.enabled:
            pareto_points = [i for i in range(len(values)) if pareto_indices[i]]
            datasets.append({'label': 'Pareto frontier',
                             'color': '#ffcc00',
                             'values': pareto_points})

        # Objectives
        for iobjective, near_indices in all_near_indices.items():
            objective = self.objectives[iobjective]
            if objective.name:
                label = objective.name
            else:
                label = 'Near Values ' + str(iobjective)
            near_points = [i for i in range(len(values)) if i in near_indices]
            near_dataset = {'label': label,
                            'color': None,
                            'values': near_points}
            datasets.append(near_dataset)

        # Dominated points
        dominated_points = [i for i in range(len(values))
                            if (self.pareto_settings.enabled and not pareto_indices[i]
                                or not self.pareto_settings.enabled)]
        datasets.append({'label': 'Dominated points',
                         'color': "#99b4d6",
                         'values': dominated_points})

        # Displays
        displays = [{'angular_component': 'results',
                     'filters': filters,
                     'datasets': datasets,
                     'values': values,
                     'references_attribute': 'array'}]
        return displays

    def export_csv(self, attribute_name: str, indices: List[int], file: str):
        """
        Exports a reduced list of objects to .csv file

        :param attribute_name: Name of the attribute in which the list is stored
        :type attribute_name: str
        :param indices: List of integers that represents selected indices of object
                        in attribute_name sequence
        :type indices: [int]
        :param file: Target file
        """
        attribute = getattr(self, attribute_name)
        lines = [attribute[i] for i in indices]
        array = np.array(lines)
        data_frame = pd.DataFrame(array, columns=self.variables)
        data_frame.to_csv(file, index=False)

    def parameters(self, variables: List[str]):
        """
        Computes Parameter objects from catalog structural data

        :param variables: List of string. Names of arguments of which
                         it should create a parameter.
        :type variables: [string]

        :return: List of Parameter objects
        """
        parameters = []
        for variable in variables:
            values = self.get_values(variable)
            parameters.append(Parameter(lower_bound=min(values),
                                        upper_bound=max(values),
                                        name=variable))
        return parameters

    def get_variable_index(self, name):
        return self.variables.index(name)

    def get_values(self, variable):
        values = [self.get_value_by_name(line, variable) for line in self.array]
        return values

    def get_value_by_name(self, line, name):
        j = self.get_variable_index(name)
        value = line[j]
        return value

    def build_costs(self, pareto_settings: ParetoSettings):
        """
        Build list of costs that are used to compute Pareto frontier.

        The cost of an attribute that is to be minimized is, for each object of catalog,
        its value minus the lower bound of of its values in the whole dataset.
        On the contrary, the cost of an attribute that is to be maximised is,
        the upper_bound of the dataset for this parameter minus the value
        of the attribute of each object of the catalog.

        For a Pareto frontier of dimensions n_costs, each vectored object of the catalog
        (n_points vectored objects in the catalog) will give a numpy array of dimensions (,n_costs)

        All put together build_costs method results in a numpy array :

        :return: A(n_points, n_costs)
        """
        pareto_parameters = self.parameters(list(pareto_settings.minimized_attributes.keys()))
        costs = np.zeros((len(self.array), len(pareto_parameters)))
        for i, line in enumerate(self.array):
            for j, parameter in enumerate(pareto_parameters):
                if pareto_settings.minimized_attributes[parameter.name]:
                    value = self.get_value_by_name(line, parameter.name) - parameter.lower_bound
                else:
                    value = parameter.upper_bound - self.get_value_by_name(line, parameter.name)
                costs[(i, j)] = value
        return costs

    @classmethod
    def random_2d(cls, bounds: Dict[str, List[float]], threshold: float, end: float = 500, name='Random Set'):
        """
        This method is for dev purpose. It can be removed if needed
        """
        array = []
        variables = list(bounds.keys())
        pareto_settings = ParetoSettings({v: True for v in variables}, enabled=True)
        objectives = []
        while len(array) <= end:
            line = [(bounds[v][1]-bounds[v][0])*np.random.rand() + bounds[v][0] for v in variables]
            if line[0]*line[1] >= threshold:
                array.append(line)
        return cls(array=array, variables=variables,
                   pareto_settings=pareto_settings, objectives=objectives,
                   choice_variables=variables, name=name)

    def handle_objective(self, objective):
        ratings = []
        for line in self.array:
            values = {variable: self.get_value_by_name(line, variable) for variable in objective.coefficients}
            rating = objective.apply_individual(values)
            ratings.append(rating)
        return ratings

    def find_best_objective(self, values: Dict[str, float], settings: ObjectiveSettings = None):
        # Unordered list of variables
        variables = list(values.keys())  # !!!
        parameters = self.parameters(variables)

        # Get pareto points
        minimized = {var: True for var in variables}
        pareto_settings = ParetoSettings(minimized_attributes=minimized)
        costs = self.build_costs(pareto_settings=pareto_settings)
        pareto_indices = pareto_frontier(costs=costs)
        pareto_values = [{var: self.get_value_by_name(self.array[i], var) for var in variables}
                         for i, is_efficient in enumerate(pareto_indices) if is_efficient]

        def apply(x):
            objective = Objective.from_angles(angles=x, variables=[p.name for p in parameters])
            best_on_pareto = min([objective.apply_individual(pareto_value) for pareto_value in pareto_values])
            rating = objective.apply_individual(values)
            delta = rating - best_on_pareto
            return delta

        i = 0
        success = False
        randomized_angles = lhs(len(values) - 1, 100)
        available_angles = [[angle * 2 * math.pi for angle in angles] for angles in randomized_angles]
        while i < len(available_angles) and not success:
            res = minimize(fun=apply, x0=randomized_angles[i], method="Powell")
            i += 1
            if res.success:
                success = True
                best_objective = Objective.from_angles(angles=res.x.tolist(),
                                                       variables=[p.name for p in parameters],
                                                       settings=settings)
                best_objective.name = "Best Coefficients" + str(self.generated_best_objectives)
                self.generated_best_objectives += 1
                self.objectives.append(best_objective)
        if not res.success:
            raise ValueError("No solutions found")


def pareto_frontier(costs):
    """
    Find the pareto-efficient points

    :param catalog: Catalog object on which to apply pareto_frontier computation
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for index, cost in enumerate(costs):
        if is_efficient[index]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < cost, axis=1)
            # And keep self
            is_efficient[index] = True
    return is_efficient


def from_csv(filename: str, end: int = None, remove_duplicates: bool = False):
    """
    Generates MBSEs from given .csv file.
    """
    array = np.genfromtxt(filename, dtype=None, delimiter=',', names=True, encoding=None)
    variables = [v for v in array.dtype.fields.keys()]
    lines = []
    for i, line in enumerate(array):
        if end is not None and i >= end:
            break
        if not remove_duplicates or (remove_duplicates and line.tolist() not in lines):
            lines.append(line.tolist())
    return lines, variables
