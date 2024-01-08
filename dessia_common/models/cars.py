#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Tests for cars data. """

import importlib.resources
from dessia_common.core import DessiaFilter
from dessia_common.tests import Car, CarWithFeatures
from dessia_common.files import StringFile

ref = importlib.resources.files("dessia_common").joinpath("models/data/cars.csv")
with ref.open('rb') as csv_cars:
    stream = StringFile.from_stream(csv_cars)

    filters = [DessiaFilter(attribute="Weight", comparison_operator="lt", bound=4000)]

    # Used models
    all_cars_no_feat = Car.from_csv(stream)
    all_cars_wi_feat = CarWithFeatures.from_csv(stream)
