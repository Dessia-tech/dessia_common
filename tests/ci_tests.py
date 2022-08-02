
import os
import unittest

scripts = ['displays.py',
           'generation.py',
           'models_test.py',
           'files.py',
           'vectored_objects.py',
           'moving_object.py',
           # Workflows
           'workflow/blocks.py',
           'workflow/workflow_with_models.py',
           'workflow/to_script.py',
           'workflow/power_simulation.py',
           'workflow/forms_simulation.py',
           'workflow/workflow_state_equalities.py',
           'workflow/workflow_diverge_converge.py',
           'workflow/workflow_method_dict.py',
           'workflow/various_tests.py',
           # breakdown
           'breakdown.py',
           # Utils
           'utils/algebra.py',
           'utils/interpolation.py',
           'utils/serialization.py',
           'utils/types.py',
           'type_matching.py',
           # Unit tests
           'unit_tests.py',
           'bson_valid.py',
           ]

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))
    exec(open(script_name).read())

# This needs to be executed once all "assert-tests" have been run + once all unittests are defined
if __name__ == '__main__':
    unittest.main(verbosity=2)
