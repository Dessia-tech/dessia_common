
import os

scripts = ['workflow_with_models.py', 'unit_tests.py', 'bson_valid.py',
           'workflow/power_simulation.py'
           ]

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))
    exec(open(script_name).read())
