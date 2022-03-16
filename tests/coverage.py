#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:35:47 2021

@author: steven
"""

import json

MIN_FILE_COVERAGE = 33.3
MIN_MODULE_COVERAGE = 69.

untracked_modules = ["dessia_common/templates.py",
                     "dessia_common/utils.py",
                     "dessia_common/optimization.py",
                     'workflows/forms_simulation.py','workflows/vectored_workflow.py',
                     'models/tests.py']

print("untracked modules:", untracked_modules)

with open("coverage.json", "r") as file:
    d = json.load(file)

total_covered = d['total']['percent_covered']
print('total covered', total_covered, '%')

min_actual_coverage = 100
for file_name, data in d['files'].items():
    summary_covered = data['summary']['percent_covered']
    print(file_name, summary_covered, '%')
    if '/'.join(file_name.split('/')[-2:]) in untracked_modules:
        print(file_name, '-> in untrack list')
    else:
        if summary_covered < MIN_FILE_COVERAGE:
            msg = f'Module {file_name} is not covered enough by tests: {summary_covered}% ' \
                  f'expected minimum {MIN_FILE_COVERAGE}%'
            raise RuntimeError(msg)
        min_actual_coverage = min(min_actual_coverage, summary_covered)

if total_covered < MIN_MODULE_COVERAGE:
    msg = f'Package is not covered enough by tests: {total_covered}% expected minimum {MIN_MODULE_COVERAGE}%'
    raise RuntimeError(msg)

print(f'[Coverage] You can increase MIN_MODULE_COVERAGE to maximum {total_covered}% (actual {MIN_MODULE_COVERAGE}%)')

print(f'[Coverage] You can increase MIN_FILE_COVERAGE to maximum {min_actual_coverage}% (actual:{MIN_FILE_COVERAGE})%')
