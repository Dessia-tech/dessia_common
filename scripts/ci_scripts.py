
import unittest

scripts = [
    # Framework
    "displays.py",
    "generation.py",
    "files.py",
    "optimization.py",
    "graph.py",
    "sampling.py",
    "checks.py",
    "datatools_models.py",
    "datatools_modeler.py",
    "datatools_math.py",
    "unit_tests.py",
    "docx_writer.py",

    # Workflows
    "workflow/power_simulation.py",
    "workflow/workflow_diverge_converge.py",
    "workflow/workflow_clustering.py",
    "workflow/workflow_building.py",
    "workflow/workflow_sampling.py",
    "workflow/workflow_inputs.py",

    # Breakdown
    "breakdown.py",

    # Utils
    "utils/algebra.py",
    "utils/interpolation.py",
    "utils/serialization.py",
    "utils/helpers.py",
    "utils/diff.py",

    # Data structures
    "data-structures/1.py"
]

for script_name in scripts:
    print(f"\n## Executing script '{script_name}'.")
    with open(script_name, "r") as script_file:
        exec(script_file.read())
        print(f"Script '{script_name}' successful.")

# This needs to be executed once all "assert-tests" have been run + once all unittests are defined
if __name__ == "__main__":
    unittest.main(verbosity=3)
