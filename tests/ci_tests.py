
import os

# for some akward reason, put scripts before unittests tests
scripts = ['displays.py',
           'generation.py',
           'models_test.py',
           'files.py',
           'vectored_objects.py',
           'clustering.py',
           'heterogeneous_list.py',
           'heterogeneous_list_pareto.py',
           'moving_object.py',
           # Workflows
           'workflow/blocks.py',
           'workflow/workflow_with_models.py',
           'workflow/to_script.py',
           'workflow/power_simulation.py',
           'workflow/forms_simulation.py',
           'workflow/workflow_state_equalities.py',
           'workflow/workflow_diverge_converge.py',
           'workflow/workflow_clustering.py',
           # breakdown
           'breakdown.py',
           # Utils
           'utils/algebra.py',
           'utils/interpolation.py',
           'utils/serialization.py',
           'type_matching.py',
           # Unit tests after that
           'unit_tests.py',
           'bson_valid.py',
           'utils/types.py',
           'utils/helpers.py'
           ]

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))
    exec(open(script_name).read())
