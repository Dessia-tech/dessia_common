import pydocstyle
import os
from glob import glob
import random

file_list = filter(lambda z: not z.endswith("__init__.py"),
                   [y for x in os.walk('./dessia_common')
                    for y in glob(os.path.join(x[0], '*.py'))])

UNWATCHED_ERRORS = [
    # To be removed from unwatched errors
    'D106',
    'D201', 'D205',
    'D404', 'D405', 'D406', 'D410', 'D411', 'D413', 'D414', 'D416',
    
    # Do not watch these errors
    'D100', 'D104', 'D105', 'D107',
    'D200', 'D202', 'D203', 'D204', 'D206', 'D210', 'D212',
    'D301', 'D302',
    'D401', 'D402', 'D407', 'D408', 'D409',
    'D412', 'D415', 'D418'
]

MAX_ERROR_BY_TYPE = {
    'D100': 1,
    'D101': 90,
    'D102': 210,
    'D103': 68,
    'D104': 1,
    'D105': 1,
    'D106': 1,
    'D107': 1,

    'D200': 1,
    'D201': 1,
    'D202': 1,
    'D203': 1,
    'D204': 1,
    'D205': 1,
    'D206': 1,
    'D207': 1,
    'D208': 1,
    'D209': 1,
    'D210': 1,
    'D211': 1,
    'D212': 1,
    'D213': 2,
    'D214': 1,
    'D215': 1,

    'D300': 1,
    'D301': 1,
    'D302': 1,

    'D400': 301,
    'D401': 1,
    'D402': 1,
    'D403': 5,
    'D404': 1,
    'D405': 1,
    'D406': 1,
    'D407': 1,
    'D408': 1,
    'D409': 1,
    'D410': 1,
    'D411': 1,
    'D412': 1,
    'D413': 1,
    'D414': 1,
    'D415': 1,
    'D416': 1,
    'D417': 1,
    'D418': 1,
}

code_to_errors = {}
for error in pydocstyle.check(file_list, ignore=UNWATCHED_ERRORS):
    code_to_errors.setdefault(error.code, [])
    code_to_errors[error.code].append(error)

code_to_number = {code: len(errors) for code, errors in code_to_errors.items()}

for error_code, number_errors in code_to_number.items():
    if error_code not in UNWATCHED_ERRORS:
        max_errors = MAX_ERROR_BY_TYPE.get(error_code, 0)

        if number_errors > max_errors:
            error_detected = True
            print(f'\nFix some {error_code} errors: {number_errors}/{max_errors}')

            errors = code_to_errors[error_code]
            errors_to_show = sorted(random.sample(errors, min(30, len(errors))),
                                    key=lambda m: (m.filename, m.line))
            for error in errors_to_show:
                print(f'{error.filename} line {error.line}: {error.message}')
        elif number_errors < max_errors:
            print(f'\nYou can lower number of {error_code} to {number_errors} (actual {max_errors})')