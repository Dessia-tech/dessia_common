import pkg_resources

from dessia_common.vectored_objects import Catalog, ParetoSettings, Objective, ObjectiveSettings
from dessia_common import DessiaFilter
from dessia_common import tests, cluster

choice_args = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model']  # Ordered
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')

all_cars = tests.CarsList.from_csv(csv_cars)
all_cars.name = 'Cars dataset'

clustest = cluster.DessiaCluster(method = 'agg', hyperparameters = {'n_clusters':4}, name = 'test')
clustest.set_hyperparameters({'n_clusters': 4, 'linkage': 'average'}, distance_threshold=1, n_clusters = None)
clustest.fit(all_cars)

