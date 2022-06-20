#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilation test
"""

import sys


# =============================================================================
#  Tests
# =============================================================================

if sys.argv[1] == 'test_valid_license':
    license_should_be_valid = True
    print('Testing for valid license')
elif sys.argv[1] == 'test_unvalid_license':
    license_should_be_valid = False
    print('Testing for unvalid license')
else:
    raise ValueError('Invalid option: use either test_valid_license ot test_unvalid_license')


test_passed = license_should_be_valid
try:
    from dessia_common.core import prettyname
    prettyname('aa')
    print('No license error triggered')
except RuntimeError:
    print('License error triggered')
    if license_should_be_valid:
        print('License was expected to be valid, test failed')
        test_passed = False
    else:
        print('License was expected to be unvalid, test passed')
        test_passed = True

assert test_passed
