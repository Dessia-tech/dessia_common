"""
Tests for dessia_common.HeterogeneousList class (loadings, check_platform and plots)
"""
import random
import numpy as npy
from dessia_common.models import all_cars_wi_feat
from dessia_common.core import HeterogeneousList
from dessia_common.cluster import CategorizedList

# =============================================================================
# TEST PARETO FRONT
# =============================================================================
tol = 0.
# Uniform
coord_1 = [random.uniform(0, 0.1) for i in range(1000)]
coord_2 = [random.uniform(0.9e6, 1e6) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = HeterogeneousList.pareto_indexes(costs, tol = tol)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(costs), costs, tol = tol)

# Uniform
coord_1 = [random.uniform(0, 0.001) for i in range(1000)]
coord_2 = [random.uniform(0, 1e6) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = HeterogeneousList.pareto_indexes(costs, tol = tol)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(costs), costs, tol = tol)

# Gaussan
coord_1 = [random.gauss(50000, 1) for i in range(1000)]
coord_2 = [random.gauss(10, 1) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = HeterogeneousList.pareto_indexes(costs, tol = tol)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(costs), costs, tol = tol)

# Cars
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
costs = [[row[3], row[1]] for row in all_cars_with_features.matrix]

pareto_points = all_cars_with_features.pareto_points(costs, tol = tol)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(all_cars_wi_feat), costs, tol = tol)
try:
    transposed_costs = list(map(list,zip(*costs)))
    pareto_frontiers = HeterogeneousList.pareto_frontiers(len(all_cars_wi_feat), transposed_costs, tol = tol)
except Exception as e:
    assert(e.args[0] ==
           "costs is length 2 and the matching HeterogeneousList is length 406. They should be the same length.")

categorized_pareto = CategorizedList.from_pareto_sheets(all_cars_with_features, costs, 7)
categorized_pareto.plot()
