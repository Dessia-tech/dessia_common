#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:56:12 2020

@author: jezequel
"""

import math
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from dessia_common import DessiaObject, Parameter, is_bounded, Filter
from scipy.optimize import minimize
from pyDOE import lhs


class ParetoSettings(DessiaObject):
    """ 
    :param coeff_names: A dictionary containing the name of a variable as key and \
    a boolean for minimization as value.
    :type coeff_names: Dict[str, bool]
    :param enabled: List of strings representing ordered variables names.
    :type enabled: bool
    :param name: The name of the block.
    :type name: str
    """
    _generic_eq = True
    _ordered_attributes = ['name', 'enabled', 'minimized_attributes']

    def __init__(self, minimized_attributes: Dict[str, bool],
                 enabled: bool = True, name: str = ''):
        self.enabled = enabled
        self.minimized_attributes = minimized_attributes

        DessiaObject.__init__(self, name=name)


class ObjectiveSettings(DessiaObject):
    _generic_eq = True
    _ordered_attributes = ['name', 'enabled', 'n_near_values']

    def __init__(self, n_near_values: int = 4,
                 enabled: bool = True, name: str = ''):
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

    def __init__(self, coefficients: Dict[str, float],
                 directions: Dict[str, bool],
                 settings: ObjectiveSettings, name: str = ''):
        for variable in coefficients:
            if variable not in directions:
                msg = "Coefficient for variable {}".format(variable)
                msg += " was found but no direction is specified."
                msg += " Add {} to directions dict.".format(variable)
                raise KeyError(msg)
        self.coefficients = coefficients
        self.directions = directions
        self.coeff_names = list(coefficients.keys())
        self.settings = settings

        DessiaObject.__init__(self, name=name)

    def apply_individual(self, values: Dict[str, float]) -> float:
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
        unsigned = [v for v in signed]
        return unsigned

    @classmethod
    def from_angles(cls, angles: List[float], variables: List[str],
                    directions: Dict[str, bool],
                    settings: ObjectiveSettings = None,
                    name="Generated from angles") -> 'Objective':
        if not isinstance(angles, list) and not isinstance(angles, np.ndarray):
            angles = [angles]
        generated_coefficients = cls.coefficients_from_angles(angles=angles)
        coefficients = {}
        for i, var in enumerate(variables):
            coeff = generated_coefficients[i]
            if (directions[var] and coeff < 0)\
                    or (not directions[var] and coeff >= 0):
                coefficients[var] = -coeff
            else:
                coefficients[var] = coeff

        if settings is None:
            settings = ObjectiveSettings()

        return Objective(coefficients=coefficients, directions=directions,
                         settings=settings, name=name)


class Catalog(DessiaObject):
    """
    Defines a Catalog object that gathers a collection of VectoredObjects

    TODO Update Docstring
    :param objectives: List of objectives to apply to
                       catalog vectored objects
    :type objectives: [Objective]
    # :param objects: List of vectored objects.
    # :type objects: [VectoredObject]
    :param choice_variables: List of string. List of variable names
                             that represent choice arguments
    :type choice_variables: [str]
    :param name: Name of the catalog
    :type name: str

    """
    _generic_eq = True
    _standalone_in_db = True
    _ordered_attributes = ['name', 'pareto_settings', 'objectives']
    _non_editable_attributes = ['array', 'variables', 'choice_variables',
                                'generated_best_objectives']
    _export_formats = ['csv']
    _allowed_methods = ['find_best_objective']
    _whitelist_attributes = ['variables']

    def __init__(self, array: List[List[float]], variables: List[str],
                 pareto_settings: ParetoSettings = None,
                 objectives: List[Objective] = None,
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

        if objectives is None:
            self.objectives = []
        else:
            self.objectives = objectives

        self.generated_best_objectives = 0

    @classmethod
    def concatenate(cls, catalogs: List['Catalog'],
                    pareto_settings: ParetoSettings = None,
                    objectives: List[Objective] = None,
                    choice_variables: List[str] = None,
                    name: str = '') -> 'Catalog':
        varsets = [set(c.variables) for c in catalogs]
        var_intersection = list(set.intersection(*varsets))

        if choice_variables is not None:
            choice_variables = [var for var in choice_variables
                                if var in var_intersection]
        else:
            choice_variables = var_intersection

        array_intersection = []
        for cat in catalogs:
            indices = [cat.get_variable_index(v) for v in var_intersection]
            for line in cat.array:
                line_intersection = [v for j, v in enumerate(line)
                                     if j in indices]
                array_intersection.append(line_intersection)

        if pareto_settings is None:
            min_attrs = {var: True for var in choice_variables}
            pareto_settings = ParetoSettings(minimized_attributes=min_attrs,
                                             enabled=False)

        catalog = cls(array=array_intersection, variables=var_intersection,
                      pareto_settings=pareto_settings, objectives=objectives,
                      choice_variables=choice_variables, name=name)
        return catalog

    # def __getitem__(self, item):
    #     print('Getitem', item)
    #     return self.array[item]
    #
    # def __getattr__(self, item):
    #     if isinstance(item, (list, tuple)):
    #         return enhanced_deep_attr(self, item)w
    #     elif isinstance(item, str) and '.' in item:
    #         sequence = item.split('.')
    #         healed_sequence = []
    #         for i, attr in enumerate(sequence):
    #             try:
    #                 healed_sequence.append(int(attr))
    #             except ValueError:
    #                 if attr in self.variables:
    #                     healed_sequence.append(self.get_variable_index(attr))
    #                 else:
    #                     healed_sequence.append(attr)
    #         print(healed_sequence)
    #         return enhanced_deep_attr(self, healed_sequence)
    #     return self.__getattribute__(item)

    def generate_multiplot(self, values: Dict[str, Any] = None):
        # TOCHECK Avoid circular imports
        import plot_data
        from plot_data.colors import BLACK, LIGHTBLUE, LIGHTGREY, BLUE

        if values is None:
            values = []
            for i, line in enumerate(self.array):
                value = {}
                for variable in self.variables:
                    value[variable] = self.get_value_by_name(line, variable)
                # for objective_name, ratings in objective_ratings.items():
                #     value[objective_name] = ratings[i]
                values.append(value)

        fontsize = 12
        first_vars = self.variables[:2]
        values2d = [{key: val[key]} for key in first_vars for val in values]
        rgbs = [[192, 11, 11], [14, 192, 11], [11, 11, 192]]

        tooltip = plot_data.Tooltip(to_disp_attribute_names=self.variables,
                                    name='Tooltip')

        scatterplot = plot_data.Scatter(axis=plot_data.Axis(),
                                        tooltip=tooltip,
                                        to_disp_attribute_names=first_vars,
                                        elements=values2d,
                                        name='Scatter Plot')

        parallelplot = plot_data.ParallelPlot(disposition='horizontal',
                                              to_disp_attribute_names=self.variables,
                                              rgbs=rgbs, elements=values)
        objects = [scatterplot, parallelplot]
        sizes = [plot_data.Window(width=560, height=300),
                 plot_data.Window(width=560, height=300)]
        coords = [(0, 0), (0, 300)]
        multiplot = plot_data.MultiplePlots(elements=values, objects=objects,
                                            sizes=sizes, coords=coords,
                                            name='Results plot')
        return multiplot

    def _display_angular(self):
        """
        Configures catalog display on frontend

        :return: List of displays dictionnaries
        """


        # filters = [variable for j, variable in enumerate(self.variables)
        #            if not isinstance(self.array[0][j], str)
        #            and variable in self.choice_variables]

        # # Pareto
        # if self.pareto_is_enabled:
        #     costs = self.build_costs(self.pareto_settings)
        #     pareto_indices = pareto_frontier(costs=costs)
        # else:
        #     pareto_indices = []
        #
        # all_near_indices = {}
        # objective_ratings = {}
        # for iobjective, objective in enumerate(self.objectives):
        #     if objective.settings.enabled:
        #         ratings = self.handle_objective(objective)
        #         if objective.name and objective.name not in objective_ratings:
        #             name = objective.name
        #         else:
        #             name = 'objective_'+str(iobjective)
        #         objective_ratings[name] = ratings
        #         filters.append(name)
        #         nval = objective.settings.n_near_values
        #         near_indices = list(np.argpartition(ratings, nval)[:nval])
        #         all_near_indices[iobjective] = near_indices

        # datasets = []
        # values = []
        # for i, line in enumerate(self.array):
        #     value = {}
        #     for variable in self.variables:
        #         value[variable] = self.get_value_by_name(line, variable)
        #     # for objective_name, ratings in objective_ratings.items():
        #     #     value[objective_name] = ratings[i]
        #     values.append(value)

        # # Pareto
        # if self.pareto_is_enabled:
        #     pareto_points = [i for i in range(len(values))
        #                      if pareto_indices[i]]
        #     datasets.append({'label': 'Pareto frontier',
        #                      'color': '#ffcc00',
        #                      'values': pareto_points})

        # # Objectives
        # for iobjective, near_indices in all_near_indices.items():
        #     objective = self.objectives[iobjective]
        #     if objective.name:
        #         label = objective.name
        #     else:
        #         label = 'Near Values ' + str(iobjective)
        #     near_points = [i for i in range(len(values)) if i in near_indices]
        #     near_dataset = {'label': label,
        #                     'color': None,
        #                     'values': near_points}
        #     datasets.append(near_dataset)

        # # Dominated points
        # dominated_points = [i for i in range(len(values))
        #                     if (self.pareto_is_enabled
        #                         and not pareto_indices[i]
        #                         or not self.pareto_is_enabled)]
        # datasets.append({'label': 'Dominated points',
        #                  'color': "#99b4d6",
        #                  'values': dominated_points})

        multiplot = self.generate_multiplot()
        dict_ = multiplot.to_dict()
        dict_['references_attribute'] = 'array'

        # Displays
        displays = {"angular_component": "plot_data",
                    "data": dict_}
        # displays = [{'angular_component': 'results',
        #              'filters': filters,
        #              'datasets': datasets,
        #              'values': values,
        #              'references_attribute': 'array'}]
        return [displays]

    def filter_(self, filters: List[Filter]):
        def apply_filters(line):
            bounded = True
            i = 0
            while bounded and i < len(filters):
                filter_ = filters[i]
                variable = filter_['attribute']
                value = line[self.get_variable_index(variable)]
                bounded = is_bounded(filter_, value)
                i += 1
            return bounded

        filtered_array = list(filter(apply_filters, self.array))
        return filtered_array

    def export_csv(self, attribute_name: str, indices: List[int], file: str):
        """
        Exports a reduced list of objects to .csv file

        :param attribute_name: Name of the attribute
                               in which the list is stored
        :type attribute_name: str
        :param indices: List of integers that represents selected
                        indices of object in attribute_name sequence
        :type indices: [int]
        :param file: Target file
        """
        attribute = getattr(self, attribute_name)
        lines = [attribute[i] for i in indices]
        array = np.array(lines)
        data_frame = pd.DataFrame(array, columns=self.variables)
        data_frame.to_csv(file, index=False)

    def parameters(self, variables: List[str]) -> List[Parameter]:
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

    def get_variable_index(self, name: str) -> int:
        return self.variables.index(name)

    def get_values(self, variable: str) -> List[float]:
        values = [self.get_value_by_name(line, variable)
                  for line in self.array]
        return values

    def get_value_by_name(self, line: List[float], name: str) -> float:
        j = self.get_variable_index(name)
        value = line[j]
        return value

    @property
    def pareto_is_enabled(self) -> bool:
        pareto = self.pareto_settings
        return pareto is not None and pareto.enabled

    def build_costs(self, pareto_settings: ParetoSettings):
        """
        Build list of costs that are used to compute Pareto frontier.

        The cost of an attribute that is to be minimized is,
        for each object of catalog, its value minus the lower bound
        of its values in the whole dataset.
        On the contrary, the cost of an attribute that is to be
        maximised, is the upper_bound of the dataset for this parameter
        minus the value of the attribute of each object of the catalog.

        For a Pareto frontier of dimensions n_costs, each vectored
        bject of the catalog (n_points vectored objects in the catalog)
        will give a numpy array of dimensions (,n_costs)

        All put together build_costs method results in a numpy array :

        :return: A(n_points, n_costs)
        """
        params = list(pareto_settings.minimized_attributes.keys())
        pareto_parameters = self.parameters(params)
        costs = np.zeros((len(self.array), len(pareto_parameters)))
        for i, line in enumerate(self.array):
            for j, parameter in enumerate(pareto_parameters):
                if pareto_settings.minimized_attributes[parameter.name]:
                    value = self.get_value_by_name(line, parameter.name)\
                            - parameter.lower_bound
                else:
                    value = parameter.upper_bound\
                            - self.get_value_by_name(line, parameter.name)
                costs[(i, j)] = value
        return costs

    @classmethod
    def random_2d(cls, bounds: Dict[str, List[float]], threshold: float,
                  end: float = 500, name='Random Set'):
        """
        This method is for dev purpose. It can be removed if needed
        """
        array = []
        variables = list(bounds.keys())
        pareto_settings = ParetoSettings({v: True for v in variables},
                                         enabled=True)
        objectives = []
        while len(array) <= end:
            line = [(bounds[v][1]-bounds[v][0])*np.random.rand() + bounds[v][0]
                    for v in variables]
            if line[0]*line[1] >= threshold:
                array.append(line)
        return cls(array=array, variables=variables,
                   pareto_settings=pareto_settings, objectives=objectives,
                   choice_variables=variables, name=name)

    def handle_objective(self, objective: Objective) -> List[float]:
        ratings = []
        for line in self.array:
            values = {variable: self.get_value_by_name(line, variable)
                      for variable in objective.coefficients}
            rating = objective.apply_individual(values)
            ratings.append(rating)
        return ratings

    def find_best_objective(self, values: Dict[str, float],
                            minimized: Dict[str, bool],
                            settings: ObjectiveSettings = None):
        # Unordered list of variables
        variables = list(values.keys())  # !!!
        parameters = self.parameters(variables)
        names = [p.name for p in parameters]

        # Get pareto points
        pareto_settings = ParetoSettings(minimized_attributes=minimized)
        costs = self.build_costs(pareto_settings=pareto_settings)
        pareto_indices = pareto_frontier(costs=costs)
        pareto_values = [{var: self.get_value_by_name(self.array[i], var)
                          for var in variables}
                         for i, is_efficient in enumerate(pareto_indices)
                         if is_efficient]

        def apply(x):
            objective = Objective.from_angles(angles=x,
                                              variables=names,
                                              directions=minimized)
            best_on_pareto = min([objective.apply_individual(pareto_value)
                                  for pareto_value in pareto_values])
            rating = objective.apply_individual(values)
            delta = rating - best_on_pareto
            return delta

        i = 0
        success = False
        randomized_angles = lhs(len(values) - 1, 100)
        available_angles = [[angle * 2 * math.pi for angle in angles]
                            for angles in randomized_angles]
        while i < len(available_angles) and not success:
            res = minimize(fun=apply, x0=randomized_angles[i], method="Powell")
            i += 1
            if res.success:
                success = True
                best_objective = Objective.from_angles(angles=res.x.tolist(),
                                                       variables=names,
                                                       directions=minimized,
                                                       settings=settings)
                best_objective.name = "Best Coefficients"\
                                      + str(self.generated_best_objectives)
                self.generated_best_objectives += 1
                self.objectives.append(best_objective)
        if not success:
            raise ValueError("No solutions found")


def pareto_frontier(costs):
    """
    Find the pareto-efficient points

    :param catalog: Catalog object on which to apply pareto_frontier
                    computation
    :return: A (n_points, ) boolean array, indicating whether each point
             is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for index, cost in enumerate(costs):
        if is_efficient[index]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < cost,
                                                axis=1)
            # And keep self
            is_efficient[index] = True
    return is_efficient


def from_csv(filename: str, end: int = None, remove_duplicates: bool = False):
    """
    Generates MBSEs from given .csv file.
    """
    array = np.genfromtxt(filename, dtype=None, delimiter=',',
                          names=True, encoding=None)
    variables = [v for v in array.dtype.fields.keys()]
    lines = []
    for i, line in enumerate(array):
        if end is not None and i >= end:
            break
        if not remove_duplicates or (remove_duplicates
                                     and line.tolist() not in lines):
            lines.append(line.tolist())
    return lines, variables
