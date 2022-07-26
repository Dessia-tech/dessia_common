#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for cars data
"""
import pkg_resources
from dessia_common.vectored_objects import Catalog, ParetoSettings, Objective, ObjectiveSettings
from dessia_common import DessiaFilter
from dessia_common.tests import Car, CarWithFeatures
from dessia_common.files import StringFile

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered

minimized_attributes = {'MPG': False, 'Horsepower': True, 'Weight': True, 'Acceleration': False}
coefficients = {'Cylinders': 0, 'MPG': -0.70, 'Displacement': 0,
                'Horsepower': 0, 'Weight': 0.70, 'Acceleration': 0, 'Model': 0}

pareto_settings = ParetoSettings(minimized_attributes=minimized_attributes, enabled=True)

csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
with open(csv_cars.name, 'r', encoding='utf-8') as stream:
    string_file = StringFile("test_cars")
    string_file.write(stream.read())

catalog = Catalog.from_csv(csv_cars)
catalog.name = 'Cars dataset'
catalog.pareto_settings = pareto_settings

filters = [DessiaFilter(attribute="Weight", operator="lt", bound=4000)]

# Empty objective because this hasn't been used/tested for a while
objective_settings = ObjectiveSettings()
objective = Objective({}, {}, objective_settings)

filtered_catalog = catalog.filter_(filters)
merged_catalog = Catalog.concatenate(catalogs=[catalog, filtered_catalog])

# Used models
all_cars_no_feat = Car.from_csv(string_file)
all_cars_wi_feat = CarWithFeatures.from_csv(string_file)
