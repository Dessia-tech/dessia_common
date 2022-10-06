"""
Tests for dessia_common.DataSet class (loadings, check_platform and plots)
"""
# import json
import random
import numpy as npy
from dessia_common.models import all_cars_wi_feat
from dessia_common.datatools import DataSet, Cluster

# =============================================================================
# TEST PARETO FRONT
# =============================================================================
# Uniform
coord_1 = [random.uniform(0, 0.1) for i in range(1000)]
coord_2 = [random.uniform(0.9e6, 1e6) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = DataSet.pareto_indexes(costs)
pareto_frontiers = DataSet.pareto_frontiers(len(costs), costs)

# Uniform
coord_1 = [random.uniform(0, 0.001) for i in range(1000)]
coord_2 = [random.uniform(0, 1e6) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = DataSet.pareto_indexes(costs)
pareto_frontiers = DataSet.pareto_frontiers(len(costs), costs)

# Gaussan
coord_1 = [random.gauss(50000, 1) for i in range(1000)]
coord_2 = [random.gauss(10, 1) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = DataSet.pareto_indexes(costs)
pareto_frontiers = DataSet.pareto_frontiers(len(costs), costs)

# Cars
all_cars_with_features = DataSet(all_cars_wi_feat)
costs = [all_cars_with_features.get_attribute_values('weight'), all_cars_with_features.get_attribute_values('mpg')]
costs = list(zip(*costs))

pareto_points = all_cars_with_features.pareto_points(costs)
pareto_frontiers = DataSet.pareto_frontiers(len(all_cars_wi_feat), costs)

# With transposed costs
transposed_costs = list(zip(*costs))
pareto_frontiers = DataSet.pareto_frontiers(len(all_cars_wi_feat), transposed_costs)
assert(npy.array_equal(pareto_frontiers[0], npy.array([[1613., 34.392097264437695], [5140.0, -297.9392097264438]])))
assert(npy.array_equal(pareto_frontiers[1], npy.array([[0.0, 1928.0], [46.6, 1508.6]])))

categorized_pareto = Cluster.from_pareto_sheets(all_cars_with_features, costs, 7)
pareto_plot_data = categorized_pareto.plot_data()
# assert(json.dumps(pareto_plot_data[0].to_dict())[150:200] == ', "Cluster Label": 0}, {"mpg": 0.0, "displacement"')
# assert(json.dumps(pareto_plot_data[1].to_dict())[10500:10550] == 't": 2901.0, "acceleration": 16.0, "Cluster Label":')
# assert(json.dumps(pareto_plot_data[2].to_dict())[50:100] == 'te_names": ["Index of reduced basis vector", "Sing')

costs = all_cars_with_features.matrix
categorized_pareto = Cluster.from_pareto_sheets(all_cars_with_features, costs, 1)
pareto_plot_data = categorized_pareto.plot_data()
# assert(json.dumps(pareto_plot_data[0].to_dict())[150:200] == ' "Cluster Label": 0}, {"mpg": 14.0, "displacement"')
# assert(json.dumps(pareto_plot_data[1].to_dict())[10500:10550] == 'acceleration": 8.5, "Cluster Label": 1}, {"mpg": 0')
# assert(json.dumps(pareto_plot_data[2].to_dict())[50:100] == 'te_names": ["Index of reduced basis vector", "Sing')

# Missing tests after coverage report
try:
    Cluster.from_pareto_sheets(all_cars_with_features, [[1,2,3]], 1)
    raise ValueError("len(costs) must be the length of corresponding DataSet")
except Exception as e:
    assert(e.args[0] == "costs is length 1 and the matching DataSet is length 406. " +
           "They should be the same length.")


