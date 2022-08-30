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
# Uniform
coord_1 = [random.uniform(0, 0.1) for i in range(1000)]
coord_2 = [random.uniform(0.9e6, 1e6) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = HeterogeneousList.pareto_indexes(costs)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(costs), costs)

# Uniform
coord_1 = [random.uniform(0, 0.001) for i in range(1000)]
coord_2 = [random.uniform(0, 1e6) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = HeterogeneousList.pareto_indexes(costs)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(costs), costs)

# Gaussan
coord_1 = [random.gauss(50000, 1) for i in range(1000)]
coord_2 = [random.gauss(10, 1) for i in range(1000)]
costs = npy.array([coord_1, coord_2]).T

pareto_points = HeterogeneousList.pareto_indexes(costs)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(costs), costs)

# Cars
all_cars_with_features = HeterogeneousList(all_cars_wi_feat)
costs = [all_cars_with_features.get_attribute_values('weight'), all_cars_with_features.get_attribute_values('mpg')]
costs = list(zip(*costs))

pareto_points = all_cars_with_features.pareto_points(costs)
pareto_frontiers = HeterogeneousList.pareto_frontiers(len(all_cars_wi_feat), costs)
try:
    transposed_costs = list(zip(*costs))
    pareto_frontiers = HeterogeneousList.pareto_frontiers(len(all_cars_wi_feat), transposed_costs)
except Exception as e:
    assert(e.args[0] ==
           "costs is length 2 and the matching HeterogeneousList is length 406. They should be the same length.")

categorized_pareto = CategorizedList.from_pareto_sheets(all_cars_with_features, costs, 7)
categorized_pareto.plot()

costs = all_cars_with_features.matrix
categorized_pareto = CategorizedList.from_pareto_sheets(all_cars_with_features, costs, 1)
categorized_pareto.plot()
