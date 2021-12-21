
import os

# for some akward reason, put scripts before unittests tests  
scripts = ['displays.py',
           'models_test.py',
           'workflow_with_models.py',
           'workflow/power_simulation.py',
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
