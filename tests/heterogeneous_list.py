"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
from dessia_common.models import all_cars_no_feat, all_cars_wi_feat, rand_data_large
from dessia_common.core import HeterogeneousList

# When attribute _features is not specified in class Car
all_cars_without_features = HeterogeneousList(all_cars_no_feat)
# When attribute _features is specified in class CarWithFeatures
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
# Auto-generated heterogeneous dataset with nb_clusters clusters of points in nb_dims dimensions
RandData_heterogeneous = HeterogeneousList(rand_data_large)

# Test on auto-generated attributes
car_matrix_with = all_cars_with_features.matrix
car_matrix_without = all_cars_without_features.matrix
heter_matrix = RandData_heterogeneous.matrix
print("car_matrix_with : \n",
      "    - n_rows", len(car_matrix_with), "\n",
      "    - n_cols", len(car_matrix_with[0]), "\n",
      "    - common_attributes", all_cars_with_features.common_attributes, "\n")
print("car_matrix_without : \n",
      "    - n_rows", len(car_matrix_without), "\n",
      "    - n_cols", len(car_matrix_without[0]), "\n",
      "    - common_attributes", all_cars_without_features.common_attributes, "\n")
print("RandData_heterogeneous : \n",
      "    - n_rows", len(heter_matrix), "\n",
      "    - n_cols", len(heter_matrix[0]), "\n",
      "    - common_attributes", RandData_heterogeneous.common_attributes, "\n")


# Tests for plot_data
all_cars_with_features.plot()
all_cars_without_features.plot()
RandData_heterogeneous.plot()

# Check platform for datasets
all_cars_with_features._check_platform()
all_cars_without_features._check_platform()
RandData_heterogeneous._check_platform()

