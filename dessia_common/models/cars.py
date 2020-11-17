#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:00:35 2020

@author: jezequel
"""

from dessia_common.vectored_objects import Catalog, Objective, ParetoSettings,\
    ObjectiveSettings, from_csv
import os

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
               'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True,
                        'Weight': True, 'Acceleration': False}
coefficients = {'Cylinders': 0, 'MPG': -0.70,  'Displacement': 0,
                'Horsepower': 0, 'Weight': 0.70, 'Acceleration': 0, 'Model': 0}

pareto_settings = ParetoSettings(minimized_attributes=minimized_attributes,
                                 enabled=True)

# objective_settings0 = ObjectiveSettings(n_near_values=4, enabled=True)
# objective0 = Objective(coefficients=coefficients,
#                        settings=objective_settings0,
#                        name='ObjectiveName')

dirname = os.path.dirname(__file__)
relative_filepath = './data/cars.csv'
filename = os.path.join(dirname, relative_filepath)

array, variables = from_csv(filename=filename, end=None,
                            remove_duplicates=True)

catalog_array = array[:10]
catalog = Catalog(array=catalog_array, variables=variables,
                  choice_variables=choice_args, objectives=[],
                  pareto_settings=pareto_settings, name='Cars')

reduced_variables = variables[:3]
reduced_array = [line[:3] for line in array[:10]]
reduced_pareto = ParetoSettings(minimized_attributes={}, enabled=False)

reduced_catalog = Catalog(array=reduced_array, variables=reduced_variables,
                          choice_variables=['MPG', 'Cylinders'],
                          objectives=[], pareto_settings=reduced_pareto,
                          name='Reduced cars')

joined_catalog = Catalog.concatenate([catalog, reduced_catalog])

# from dessia_api_client import Client
# c = Client(api_url = 'https://api.platform-dev.dessia.tech')
# r = c.create_object_from_python_object(reduced_catalog)
