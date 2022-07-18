"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import pkg_resources
import random
import matplotlib.pyplot as plt
import numpy as npy
from dessia_common import tests
from dessia_common.core import HeterogeneousList

# Standard cars homogeneous dataset from the Internet
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(tests.Car.from_csv(csv_cars))
# When attribute _features is specified in class CarWithFeatures
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
all_cars_with_features = HeterogeneousList(tests.CarWithFeatures.from_csv(csv_cars))

# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
clustesters_heterogeneous = HeterogeneousList(tests.ClusTesterD9.create_dataset(nb_clusters=10, nb_points=500) +
                                              tests.ClusTesterD7.create_dataset(nb_clusters=10, nb_points=500) +
                                              tests.ClusTesterD8.create_dataset(nb_clusters=10, nb_points=500))

# Test on auto-generated attributes
car_matrix_with = all_cars_with_features.matrix
car_matrix_without = all_cars_without_features.matrix
heter_matrix = clustesters_heterogeneous.matrix
print("car_matrix_with : \n",
      "    - n_rows", len(car_matrix_with), "\n",
      "    - n_cols", len(car_matrix_with[0]), "\n",
      "    - common_attributes", all_cars_with_features.common_attributes, "\n")
print("car_matrix_without : \n",
      "    - n_rows", len(car_matrix_without), "\n",
      "    - n_cols", len(car_matrix_without[0]), "\n",
      "    - common_attributes", all_cars_without_features.common_attributes, "\n")
print("clustesters_heterogeneous : \n",
      "    - n_rows", len(heter_matrix), "\n",
      "    - n_cols", len(heter_matrix[0]), "\n",
      "    - common_attributes", clustesters_heterogeneous.common_attributes, "\n")


# Tests for plot_data
all_cars_with_features.plot()
all_cars_without_features.plot()
clustesters_heterogeneous.plot()

# Check platform for datasets
all_cars_with_features._check_platform()
all_cars_without_features._check_platform()
clustesters_heterogeneous._check_platform()


# =============================================================================
# TEST PARETO FRONT
# =============================================================================

# Uniform
coord_1 = [random.uniform(0, 10) for i in range(1000)]
coord_2 = [random.uniform(0, 10) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_frontiers = HeterogeneousList.pareto_frontiers(costs)

# Gaussan
coord_1 = [random.gauss(5, 1) for i in range(1000)]
coord_2 = [random.gauss(5, 1) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_frontiers = HeterogeneousList.pareto_frontiers(costs)

# Cars
costs = npy.array([[row[4], row[1]] for row in all_cars_with_features.matrix])
pareto_frontiers = HeterogeneousList.pareto_frontiers(costs)


