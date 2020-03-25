#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:56:12 2020

@author: jezequel
"""

from typing import List
import numpy as np
import pandas as pd
from dessia_common import DessiaObject, Parameter


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

    def __init__(self, coeff_names: List[str], coeff_values: List[float],
                 scaled: bool = False, name: str = ''):
        self.coeff_names = coeff_names
        self.coeff_values = coeff_values
        # self.coeff_min = coeff_min

        self.scaled = scaled

        DessiaObject.__init__(self, name=name)

    def apply_to_catalog(self, catalog):
        parameters = catalog.parameters(self.coeff_names)
        ratings = []
        means = catalog.means([p.name for p in parameters])
        for line in catalog.array:
            values = [catalog.get_value_by_name(line, p.name) for p in parameters]
            objective = sum([v * c / m if self.scaled else v * c for v, c, m in zip(values, self.coeff_values, means)])
            ratings.append(objective)
        return ratings


class Catalog(DessiaObject):
    """
    Defines a Catalog object that gathers a collection of VectoredObjects

    :param pareto_attributes: List of strings representing names of variables
                              used for pareto computations.
    :type pareto_attributes: [str]
    :param minimise: List of booleans representing if pareto for this variable
                     should be searched in maximum or minimum direction.
    :type minimise: [bool]
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
    _non_editable_attributes = ['array', 'variables', 'choice_variables']
    _export_formats = ['csv']

    def __init__(self, array: List[List[float]], variables: List[str],
                 pareto_attributes: List[str], minimise: List[bool],
                 objectives: List[Objective], n_near_values: int,
                 choice_variables: List[str] = None,
                 enable_pareto: bool = True, enable_objectives: bool = True,
                 name: str = ''):
        DessiaObject.__init__(self, name=name)

        self.array = array
        self.variables = variables
        self.choice_variables = choice_variables

        self.pareto_attributes = pareto_attributes
        self.minimise = minimise

        self.objectives = objectives
        self.n_near_values = n_near_values

        self.enable_pareto = enable_pareto
        self.enable_objectives = enable_objectives

    def _display_angular(self):
        """
        Configures catalog display on frontend

        :return: List of displays dictionnaries
        """
        filters = [{'attribute': variable, 'operator': 'gt', 'bound': 0} for j, variable in enumerate(self.variables)
                   if not isinstance(self.array[0][j], str) and variable in self.choice_variables]

        values = [{f['attribute']: self.get_value_by_name(line, f['attribute']) for f in filters}
                  for line in self.array]

        # Pareto
        pareto_indices = pareto_frontier(catalog=self)

        all_near_indices = []
        already_used = []
        if self.enable_objectives:
            for objective in self.objectives:
                ratings = objective.apply_to_catalog(self)
                indexed_ratings = [(i, r) for i, r in enumerate(ratings)]
                count = 0
                near_indices = []
                while count < self.n_near_values and indexed_ratings:
                    min_index = indexed_ratings.index(min(indexed_ratings, key=lambda t: t[1]))
                    min_tuple = indexed_ratings.pop(min_index)
                    if min_tuple[0] not in already_used:
                        near_indices.append(min_tuple[0])
                        already_used.append(min_tuple[0])
                        count += 1
                all_near_indices.append(near_indices)

        # Dominated points
        dominated_points = [i for i in range(len(values)) if not pareto_indices[i] and i not in already_used]
        dominated_values = [(i, value) for i, value in enumerate(values)
                            if not pareto_indices[i] and i not in already_used]
        datasets = [{'label': 'Dominated points',
                     'color': "#99b4d6",
                     'values': dominated_points}]

        # Pareto
        if self.enable_pareto:
            pareto_points = [i for i in range(len(values)) if pareto_indices[i]]
            pareto_values = [(i, value) for i, value in enumerate(values)
                             if pareto_indices[i] and i not in already_used]
            datasets.append({'label': 'Pareto frontier',
                             'color': '#ffcc00',
                             'values': pareto_points})

        # Objectives
        if self.enable_objectives:
            near_points = [[i for i in range(len(values)) if i in near_indices]
                           for near_indices in all_near_indices]
            near_values = [[(i, value) for i, value in enumerate(values) if i in near_indices]
                           for near_indices in all_near_indices]
            near_datasets = [{'label': 'Near Values ' + str(i),
                              'color': None,
                              'values': nv} for i, nv in enumerate(near_points)]
            datasets.extend(near_datasets)

        # Displays
        displays = [{'angular_component': 'results',
                     'filters': filters,
                     'datasets': datasets,
                     'values': values,
                     'references_attribute': 'objects'}]
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
        if self._init_variables is not None:
            attributes = list(self._init_variables.keys())  # !!! Unordered
            attribute = getattr(self, attribute_name)
            lines = [attribute[i].to_array() for i in indices]
            array = np.array(lines)
            data_frame = pd.DataFrame(array, columns=attributes)
            data_frame.to_csv(file, index=False)
        msg = 'Class {} should implement _init_variables'.format(self.__class__)
        msg += ' in order to be exportable'
        raise ValueError(msg)

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

    def get_values(self, variable):
        values = [self.get_value_by_name(line, variable) for line in self.array]
        return values

    def get_value_by_name(self, line, name):
        j = self.variables.index(name)
        value = line[j]
        return value

    def mean(self, variable):
        values = self.get_values(variable)
        mean = sum(values)/len(values)
        return mean

    def means(self, variables: List[str]):
        means = []
        for variable in variables:
            means.append(self.mean(variable))
        return means

    def build_costs(self):
        """
        Build list of costs that are used to compute Pareto frontier.

        The cost of an attribute that is to be minimised is, for each object of catalog,
        its value minus the lower bound of of its values in the whole dataset.
        On the contrary, the cost of an attribute that is to be maximised is,
        the upper_bound of the dataset for this parameter minus the value
        of the attribute of each object of the catalog.

        For a Pareto frontier of dimensions n_costs, each vectored object of the catalog
        (n_points vectored objects in the catalog) will give a numpy array of dimensions (,n_costs)

        All put together build_costs method results in a numpy array :

        :return: A(n_points, n_costs)
        """
        pareto_parameters = self.parameters(self.pareto_attributes)
        costs = np.zeros((len(self.array), len(pareto_parameters)))
        for i, line in enumerate(self.array):
            for j, parameter in enumerate(pareto_parameters):
                if self.minimise[j]:
                    value = self.get_value_by_name(line, parameter.name) - parameter.lower_bound
                else:
                    value = parameter.upper_bound - self.get_value_by_name(line, parameter.name)
                costs[(i, j)] = value
        return costs


def pareto_frontier(catalog: Catalog):
    """
    Find the pareto-efficient points

    :param catalog: Catalog object on which to apply pareto_frontier computation
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = catalog.build_costs()
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
