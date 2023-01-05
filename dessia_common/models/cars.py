#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for cars data.
"""
import pkg_resources
from dessia_common.core import DessiaFilter
from dessia_common.tests import Car, CarWithFeatures
from dessia_common.files import StringFile

csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
stream = StringFile.from_stream(csv_cars)

csv_cars.seek(0)

filters = [DessiaFilter(attribute="Weight", comparison_operator="lt", bound=4000)]

# Used models
all_cars_no_feat = Car.from_csv(stream)
all_cars_wi_feat = CarWithFeatures.from_csv(stream)
