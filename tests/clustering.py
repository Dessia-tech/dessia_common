"""
Cluster.py package testing.
"""
import pkg_resources
from dessia_common import tests, cluster
import numpy as np

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')

all_cars, variables = tests.Car.from_csv(csv_cars)
cars_matrix = [car.to_vector() for car in all_cars]


dbtest = cluster.ClusterResult.from_dbscan(all_cars, eps=0.1)
aggclustest = cluster.ClusterResult.from_agglomerative_clustering(cars_matrix, n_clusters=10)
kmeanstest = cluster.ClusterResult.from_kmeans(all_cars, n_clusters=10)


db_list = dbtest.data_to_clusters(all_cars, dbtest.labels)
agg_list = aggclustest.data_to_clusters(all_cars, aggclustest.labels)
# kmeans_list = kmeanstest.data_to_clusters(all_cars, kmeanstest.labels)

# dbtest.check_dimensionality(all_cars)
# aggclustest.check_dimensionality(all_cars)
# kmeanstest.check_dimensionality(all_cars)

dbtest.plot_data()
aggclustest.plot_data()
# kmeanstest.plot_data()
