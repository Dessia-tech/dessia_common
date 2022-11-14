"""
Tests for dessia_common.modeling file

"""

from dessia_common.core import DessiaObject
from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.datatools.modeling import DessiaScaler, StandardScaler, IdentityScaler, LinearRegression


# Load Data and put it in a Dataset (matrix is automatically computed)
dataset_example = Dataset(all_cars_no_feat)

# Test scalers
idty_scaler = IdentityScaler().fit(dataset_example.matrix)
idty_matrix = idty_scaler.transform(dataset_example.matrix)
idty_scaler, idty_matrix = idty_scaler.fit_transform(dataset_example.matrix)

std_scaler = StandardScaler().fit(dataset_example.matrix)
std_matrix = std_scaler.transform(dataset_example.matrix)
std_scaler, std_matrix = std_scaler.fit_transform(dataset_example.matrix)

# Tests models
linear_model = LinearRegression().fit(std_matrix)


