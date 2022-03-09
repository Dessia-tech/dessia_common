from dessia_common.models.cars import catalog, merged_catalog
from dessia_common.vectored_objects import pareto_frontier

costs = catalog.build_costs(pareto_settings=catalog.pareto_settings)
pareto = pareto_frontier(costs)

assert sum(pareto) == 24
assert len(merged_catalog.array) == 745
