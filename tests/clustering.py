import pkg_resources
import numpy as npy
from io import StringIO

from dessia_common.vectored_objects import Catalog, ParetoSettings, Objective, ObjectiveSettings
from dessia_common import DessiaFilter
from dessia_common import tests, cluster

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')

all_cars, variables = tests.Car.from_csv(csv_cars)

clustest = cluster.ClusterResult.fromAgglomerativeClustering(all_cars, n_clusters = 10)
clusters_list = clustest.data_to_clusters(clustest.data, clustest.labels)



