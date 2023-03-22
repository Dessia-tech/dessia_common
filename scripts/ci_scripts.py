
import unittest

scripts = [
    # Framework
    "displays.py",
    "generation.py",
    "models_test.py",
    "files.py",
    "clustering.py",
    "dataset.py",
    "dataset_pareto.py",
    "filters.py",
    "moving_object.py",
    "optimization.py",
    "graph.py",
    "sampling.py",
    "markdowns.py",
    "checks.py",
    "unit_tests.py",

    # Workflows
    "workflow/blocks.py",
    "workflow/workflow_with_models.py",
    "workflow/power_simulation.py",
    "workflow/forms_simulation.py",
    "workflow/pipes.py",
    "workflow/workflow_state_equalities.py",
    "workflow/workflow_diverge_converge.py",
    "workflow/workflow_clustering.py",
    "workflow/workflow_filtering.py",
    "workflow/workflow_pareto.py",
    "workflow/workflow_building.py",
    "workflow/workflow_sampling.py",
    "workflow/workflow_inputs.py",
    "workflow/file_inputs.py",

    # Breakdown
    "breakdown.py",

    # Utils
    "utils/algebra.py",
    "utils/interpolation.py",
    "utils/serialization.py",
    "utils/helpers.py",
    "utils/diff.py",
    "type_matching.py",

    # Data structures
    "data-structures/1.py"
]

for script_name in scripts:
    print(f"\n## Executing script '{script_name}'.")
    exec(open(script_name).read())
    print(f"Script '{script_name}' successful.")

# This needs to be executed once all "assert-tests" have been run + once all unittests are defined
if __name__ == "__main__":
    unittest.main(verbosity=3)