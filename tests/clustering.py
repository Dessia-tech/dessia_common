import pkg_resources
import numpy as npy
from io import StringIO

from dessia_common.vectored_objects import Catalog, ParetoSettings, Objective, ObjectiveSettings
from dessia_common import DessiaFilter
from dessia_common import tests, cluster

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')

all_cars, variables = tests.Car.from_csv(csv_cars)

dbtest = cluster.ClusterResult.fromDBSCAN(all_cars, eps = 15)
aggclustest = cluster.ClusterResult.fromAgglomerativeClustering(all_cars, n_clusters = 10)
kmeanstest = cluster.ClusterResult.fromKMeans(all_cars, n_clusters = 10)


db_list = dbtest.data_to_clusters(all_cars, dbtest.labels)
agg_list = aggclustest.data_to_clusters(all_cars, aggclustest.labels)
kmeans_list = kmeanstest.data_to_clusters(all_cars, kmeanstest.labels)



