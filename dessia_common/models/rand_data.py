#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models for RandData objects data.
"""

from dessia_common.tests import RandDataD1, RandDataD2, RandDataD3, RandDataD4, RandDataD5, RandDataD6, RandDataD7,\
    RandDataD8, RandDataD9, RandDataD10

# Tests RandDatas
mean_borns = (-50, 50)
std_borns = (-2, 2)

rand_data_d1 = RandDataD1.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d2 = RandDataD2.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d3 = RandDataD3.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d4 = RandDataD4.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d5 = RandDataD5.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d6 = RandDataD6.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d7 = RandDataD7.create_dataset(nb_clusters=10, nb_points=500, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d8 = RandDataD8.create_dataset(nb_clusters=10, nb_points=500, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d9 = RandDataD9.create_dataset(nb_clusters=10, nb_points=500, mean_borns=mean_borns, std_borns=std_borns)
rand_data_d10 = RandDataD10.create_dataset(nb_clusters=10, nb_points=250, mean_borns=mean_borns, std_borns=std_borns)

rand_data_small = rand_data_d5 + rand_data_d4 + rand_data_d3
rand_data_middl = rand_data_d6 + rand_data_d4 + rand_data_d5
rand_data_large = rand_data_d9 + rand_data_d7 + rand_data_d8
