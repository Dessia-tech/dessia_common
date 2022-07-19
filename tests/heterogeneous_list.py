"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import random
import pkg_resources
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

# Test on matrices
idx = random.randint(0, len(all_cars_with_features) - 1)
assert(all(item in all_cars_with_features.matrix[idx]
           for item in [getattr(all_cars_with_features.dessia_objects[idx], attr)
                        for attr in all_cars_with_features.common_attributes]))

idx = random.randint(0, len(all_cars_without_features) - 1)
assert(all(item in all_cars_without_features.matrix[idx]
           for item in [getattr(all_cars_without_features.dessia_objects[idx], attr)
                        for attr in all_cars_without_features.common_attributes]))

idx = random.randint(0, len(clustesters_heterogeneous) - 1)
assert(all(item in clustesters_heterogeneous.matrix[idx]
           for item in [getattr(clustesters_heterogeneous.dessia_objects[idx], attr)
                        for attr in clustesters_heterogeneous.common_attributes]))



# Tests for plot_data
# all_cars_with_features.plot()
# all_cars_without_features.plot()
# clustesters_heterogeneous.plot()

# # # Check platform for datasets
# all_cars_with_features._check_platform()
# all_cars_without_features._check_platform()
# clustesters_heterogeneous._check_platform()

# Check for sorts
print(all_cars_with_features)
print(all_cars_with_features[0:10])
print(all_cars_with_features[2])
print(all_cars_with_features[:10])
all_cars_with_features.sort('weight', ascend=False)
print(all_cars_with_features[:10])
all_cars_without_features.sort(0)
print(f"sort name : {all_cars_without_features.common_attributes[0]}")
print(all_cars_without_features[:10])

# Filters
print("Filters tests")
# print(all_cars_with_features[ all_cars_with_features.tolist(attr) == 70 ])


