#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:00:35 2020

@author: jezequel
"""

from dessia_common.vectored_objects import Catalog, Objective, ParetoSettings,\
    ObjectiveSettings, from_csv, pareto_frontier
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
#                         settings=objective_settings0,
#                         name='ObjectiveName')

dirname = os.path.dirname(__file__)
relative_filepath = './data/cars.csv'
filename = os.path.join(dirname, relative_filepath)

array, variables = from_csv(filename=filename, end=None,
                            remove_duplicates=True)

catalog = Catalog(array=array, variables=variables,
                  choice_variables=choice_args, objectives=[],
                  pareto_settings=pareto_settings, name='Cars')

# from dessia_api_client import Client
# c = Client(api_url='https://api.platform-dev.dessia.tech')
# r = c.create_object_from_python_object(catalog)


from plot_data import plot_canvas

plot_canvas(plot_data_object=catalog.plot_data()[0], canvas_id='canvas')

# cost = catalog.build_costs(pareto_settings)
# print('cost', cost)

# p_f = pareto_frontier(cost)
# print('p_f', p_f)