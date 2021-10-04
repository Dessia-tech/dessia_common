
import os

scripts = ['unit_tests.py', 'bson_valid.py'
]

#  'cyl_cyl.py', 'cyl_pf.py',
# 'ls3D_ls3D.py', 'sweep_sweep.py', 'tore_cyl.py','tore_pf.py'
# 'tore_tore.py'

for script_name in scripts:
    print('\n## Executing script {}'.format(script_name))

    exec(open(script_name).read())
