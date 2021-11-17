
import os

scripts = ['displays.py',
           'workflow_with_models.py',
           'workflow/power_simulation.py',
           'unit_tests.py', 'bson_valid.py',
           'utils/algebra.py'
           ]

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))
    exec(open(script_name).read())
