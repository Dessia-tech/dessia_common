#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:56:12 2020

@author: jezequel
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from dessia_common import DessiaObject, Parameter

class VectoredObject(DessiaObject):
    """
    Defines a "vectored object". This is a vectored representation of a structured data value.
    It is the MBSE view of a generically structured object.

    :param name: Name of instance.
    :type name: str
    :param kwargs: Dictionnary that associates variable names to their values for this instance.
    """
    _generic_eq = True
    _standalone_in_db = True
    def __init__(self, name: str = '', **kwargs):
        DessiaObject.__init__(self, name=name, **kwargs)

    def vectorize(self, parameters:List[Parameter], coefficients:List[float], bounds:List[Tuple[float, float]]):
        """
        Computes an adimensionnal vector representing the vectored object.

        :param parameters: List of Parameter objects
        :type parameters: [Parameter]

        :return: Vector of dimension n, n being the dimension of the structural data
        """
        vector = []
        if coefficients is None:
            coefficients = [1]*len(parameters)
        for parameter, coefficient, interval in zip(parameters, coefficients, bounds):
            value = getattr(self, parameter.name)
            if (interval[0] is not None and value < interval[0])\
                or (interval[1] is not None and value > interval[1]):
                return None
            normalized_value = coefficient*parameter.normalize(value)
            vector.append(normalized_value)
        return vector

    def to_array(self):
        """
        TODO
        """
        values = [getattr(self, name) for name in self._init_variables]
        return values


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
    def __init__(self, coeff_names: List[str], coeff_values: List[float], name: str = ''):
        self.coeff_names = coeff_names
        self.coeff_values = coeff_values

        DessiaObject.__init__(self, name=name)

    def apply(self, vectored_object: VectoredObject):
        """
        Applies objective function to given vectored object

        :param vectored_object: Vectored object on which to apply objective
        :type vectored_object: VectoredObject

        :return: float, Rating of given object according to this objective
        """
        objectives = [getattr(vectored_object, arg)*coefficient
                      for arg, coefficient in zip(self.coeff_names, self.coeff_values)]
        objective = sum(objectives)
        return objective


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
    :param objects: List of vectored objects.
    :type objects: [VectoredObject]
    :param choice_args: List of string. List of variable names that represent choice arguments
    :type choice_args: [str]
    :param name: Name of the catalog
    :type name: str

    """
    _generic_eq = True
    _standalone_in_db = True
    _editable_variables = ['minimise', 'pareto_attributes', 'n_near_values',
                           'objectives', 'enable_pareto', 'enable_objectives', 'name']
    _export_formats = ['csv']
    def __init__(self, pareto_attributes: List[str], minimise: List[bool],
                 objectives: List[Objective], n_near_values: int,
                 objects: List[VectoredObject] = None, choice_args: List[str] = None,
                 enable_pareto: bool = True, enable_objectives: bool = True,
                 name: str = ''):
        DessiaObject.__init__(self, name=name)

        self.objects = objects
        self.choice_args = choice_args

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
        filters = [{'attribute' : arg, 'operator' : 'gt', 'bound' : 0}\
                   for arg, value in self.objects[0].__dict__.items()\
                   if arg != 'name' and not isinstance(value, str)\
                       and not hasattr(self.__class__, arg)\
                       and arg in self.choice_args]

        values = [{f['attribute'] : getattr(o, f['attribute']) for f in filters}\
                  for o in self.objects]

        # Pareto
        pareto_indices = pareto_frontier(catalog=self)

        all_near_indices = []
        already_used = []
        if self.enable_objectives:
            for objective in self.objectives:
                ratings = [(i, objective.apply(o)) for i, o in enumerate(self.objects)]
                i = 0
                near_indices = []
                while i < self.n_near_values and ratings:
                    new_min_index = ratings.index(min(ratings, key=lambda t: t[1]))
                    new_min = ratings.pop(new_min_index)
                    if new_min[0] not in already_used:
                        near_indices.append(new_min[0])
                        already_used.append(new_min[0])
                        i += 1
                all_near_indices.append(near_indices)

        # Dominated points
        dominated_values = [(i, value) for i, value in enumerate(values)
                            if not pareto_indices[i] and i not in already_used]
        datasets = [{'label' : 'Dominated points',
                     'color' : "#99b4d6",
                     'values' : dominated_values}]

        # Pareto
        if self.enable_pareto:
            pareto_values = [(i, value) for i, value in enumerate(values)\
                             if pareto_indices[i] and i not in already_used]
            datasets.append({'label' : 'Pareto frontier',
                             'color' : '#ffcc00',
                             'values' : pareto_values})

        # Objectives
        if self.enable_objectives:
            near_values = [[(i, value) for i, value in enumerate(values) if i in near_indices]
                           for near_indices in all_near_indices]
            near_datasets = [{'label' : 'Near Values ' + str(i),
                              'color' : None,
                              'values' : nv} for i, nv in enumerate(near_values)]
            datasets.extend(near_datasets)

        # Displays
        displays = [{'angular_component': 'results',
                     'filters': filters,
                     'datasets': datasets,
                     'references_attribute': 'objects'}]
        return displays

    def export_csv(self, attribute_name:str, indices:List[int], file:str):
        """
        Exports a reduced list of objects to .csv file

        :param attibute_name: Name of the attribute in which the list is stored
        :type attribute_name: str
        :param indices: List of integers that represents selected indices of object
                        in attribute_name sequence
        :type indices: [int]
        :param file: Target file
        """
        # attributes = ['e', 'tgte', 'h', 'ref', 'nw', 'REFrlt', 'NPSHeval',
        #               'coutTotal_horsQuincaill', 'Fwrlt',
        #               'radialSpringConfidenceRatio', 'Sk', 'contrainte', 'Sp']
        attributes = list(self._init_variables.keys()) # !!! Unordered
        attribute = getattr(self, attribute_name)
        lines = [attribute[i].to_array() for i in indices]
        array = np.array(lines)
        data_frame = pd.DataFrame(array, columns=attributes)
        data_frame.to_csv(file, index=False)


    def parameters(self, argnames:List[str]):
        """
        Computes Parameter objects from catalog structural data

        :param argnames: List of string. Names of arguments of which
                         it should create a parameter.
        :type argnames: [string]

        :return: List of Parameter objects
        """
        parameters = []
        for arg in argnames:
            values = [getattr(o, arg) for o in self.objects]
            parameters.append(Parameter(lower_bound=min(values),
                                        upper_bound=max(values),
                                        name=arg))
        return parameters

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
        costs = np.zeros((len(self.objects), len(pareto_parameters)))
        for i, object_ in enumerate(self.objects):
            for j, parameter in enumerate(pareto_parameters):
                if self.minimise[j]:
                    value = getattr(object_, parameter.name) - parameter.lower_bound
                else:
                    value = parameter.upper_bound - getattr(object_, parameter.name)
                costs[(i, j)] = value
        return costs

    def apply_objective(self, objective:Objective):
        """
        Given an Objective object, applies it to every object of the catalog.

        :return: List of float. Ratings to the applied objective
        """
        ratings = [objective.apply(o) for o in self.objects]
        return ratings


def pareto_frontier(catalog:Catalog):
    """
    Find the pareto-efficient points

    :param costs: An (n_points, n_costs) array
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

def from_csv(filename:str, class_:type=VectoredObject, end:int=None, remove_duplicates:bool=False):
    """
    Generates MBSEs from given .csv file.
    """
    array = np.genfromtxt(filename, dtype=None, delimiter=',', names=True)

    objects = []
    for i, line in enumerate(array):
        if end is not None and i >= end:
            break
        kwargs = {}
        for variable_name in array.dtype.fields.keys():
            pyval = line[variable_name].item()
            if isinstance(pyval, bytes):
                pyval = str(pyval)
            kwargs[variable_name] = pyval
        object_ = class_(name='object' + str(i), **kwargs)
        if not remove_duplicates or (remove_duplicates and object_ not in objects):
            objects.append(object_)
    return objects