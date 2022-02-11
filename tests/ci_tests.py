
import os

# for some akward reason, put scripts before unittests tests  
scripts = ['displays.py',
           'generation.py',
           'models_test.py',
           'files.py',
           # Workflows
           'workflow_with_models.py',
           'workflow/power_simulation.py',
           'workflow/forms_simulation.py',
           'workflow_state_equalities.py',
           # Utils
           'utils/algebra.py',
           'utils/interpolation.py',
           'utils/serialization.py',
           # Unit tests after that
           'unit_tests.py',
           'bson_valid.py',
           ]

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))
    exec(open(script_name).read())
