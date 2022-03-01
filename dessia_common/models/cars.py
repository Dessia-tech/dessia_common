#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:00:35 2020

@author: jezequel
"""

import pkg_resources

from dessia_common.vectored_objects import Catalog, ParetoSettings

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
               'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True, 'Weight': True, 'Acceleration': False}
coefficients = {'Cylinders': 0, 'MPG': -0.70, 'Displacement': 0,
                'Horsepower': 0, 'Weight': 0.70, 'Acceleration': 0, 'Model': 0}

pareto_settings = ParetoSettings(minimized_attributes=minimized_attributes, enabled=True)

csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')

catalog = Catalog.from_csv(csv_cars)
catalog.name = 'Cars dataset'
catalog.pareto_settings = pareto_settings
