"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import pkg_resources
from dessia_common import tests
from dessia_common.core import HeterogeneousList

# Standard cars homogeneous dataset from the Internet
csv_cars = pkg_resources.resource_stream('dessia_common', 'models/data/cars.csv')
all_cars_homomgeneous = HeterogeneousList(tests.Car.from_csv(csv_cars))

# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
clustesters_heterogeneous = HeterogeneousList(tests.ClusTester_d9.create_dataset(nb_clusters=10, nb_points=500) +
                                              tests.ClusTester_d7.create_dataset(nb_clusters=10, nb_points=500) +
                                              tests.ClusTester_d8.create_dataset(nb_clusters=10, nb_points=500))

# Test on auto-generated attributes
homog_matrix = all_cars_homomgeneous.matrix
heter_matrix = clustesters_heterogeneous.matrix
print("all_cars_homomgeneous : \n",
      "    - n_rows", len(homog_matrix), "\n",
      "    - n_cols", len(homog_matrix[0]), "\n",
      "    - common_attributes", all_cars_homomgeneous.common_attributes, "\n")
print("clustesters_heterogeneous : \n",
      "    - n_rows", len(heter_matrix), "\n",
      "    - n_cols", len(heter_matrix[0]), "\n",
      "    - common_attributes", clustesters_heterogeneous.common_attributes, "\n")

# Tests for plot_data
all_cars_homomgeneous.plot()
clustesters_heterogeneous.plot()

# Check platform for datasets
all_cars_homomgeneous._check_platform()
clustesters_heterogeneous._check_platform()
