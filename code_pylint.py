'''
Read pylint errors to see if number of errors does not exceed specified limits
v1.2

Changes:
    v1.1: move imports to top
    v1.2: limit to 100 message to avoid overflow, global note check at end, ratchet effects
'''


import os
import sys
from pylint import __version__
from pylint.lint import Run

MIN_NOTE = 9.17

UNWATCHED_ERRORS = ['fixme', 'trailing-whitespace', 'import-error']

MAX_ERROR_BY_TYPE = {
                     'protected-access': 45,
                     'invalid-name': 0,
                     'consider-using-f-string': 1,
                     'no-else-return': 0,
                     'arguments-differ': 2,
                     'no-member': 1,
                     'too-many-locals': 12,  # Reduce by dropping vectored objects
                     'wrong-import-order': 7,
                     'too-many-branches': 10,
                     'unused-import': 1,
                     'unused-argument': 5,
                     'cyclic-import': 12,
                     'unused-import': 0,
                     'unused-argument': 6,
                     'cyclic-import': 10,
                     'no-self-use': 6,
                     'unused-variable': 0,
                     'trailing-whitespace': 11,
                     'empty-docstring': 7,
                     'missing-module-docstring': 1,
                     'too-many-arguments': 16,
                     'too-few-public-methods': 5,
                     'unnecessary-comprehension': 5,
                     'no-value-for-parameter': 2,
                     'too-many-return-statements': 10,
                     'raise-missing-from': 0,
                     'consider-merging-isinstance': 1,
                     'abstract-method': 6,
                     'import-outside-toplevel': 7,  # TODO : will reduced in a future work (when tests are ready)
                     'too-many-instance-attributes': 7,
                     'consider-iterating-dictionary': 1,
                     'attribute-defined-outside-init': 3,
                     'simplifiable-if-expression': 1,
                     'broad-except': 3,
                     'consider-using-get': 0,
                     'undefined-loop-variable': 1,
                     'consider-using-with': 2,
                     'too-many-nested-blocks': 2,
                     'bad-staticmethod-argument': 1,
                     'too-many-public-methods': 2,  # Try to lower by splitting DessiaObject and Workflow
                     'consider-using-generator': 1,
                     'too-many-statements': 1,
                     'chained-comparison': 1,
                     'wildcard-import': 1,
                     'use-maxsplit-arg': 1,
                     'duplicate-code': 1,
                     # No tolerance errors
                     'eval-used': 0,
                     'redefined-builtin': 0,
                     'arguments-renamed': 0,
                     'ungrouped-imports': 0,
                     'super-init-not-called': 0,
                     'superfluous-parens': 0,
                     'unused-wildcard-import': 0,
                     'consider-using-enumerate': 0,
                     'undefined-variable': 0,
                     'function-redefined': 0,
                     'inconsistent-return-statements': 0,
                     'unexpected-special-method-signature': 0,
                     'too-many-lines': 1,
                     'bare-except': 3,
                     'unspecified-encoding': 0,
                     'no-else-raise': 0,
                     'bad-indentation': 0,
                     'reimported': 0,
                     'use-implicit-booleaness-not-comparison': 0,
                     'misplaced-bare-raise': 0,
                     'redefined-argument-from-local': 0,
                     'import-error': 0,
                     'unsubscriptable-object': 0,
                     'unnecessary-dunder-call': 0
                     }

print('pylint version: ', __version__)

f = open(os.devnull, 'w')

old_stdout = sys.stdout
sys.stdout = f

results = Run(['dessia_common', '--output-format=json', '--reports=no'], do_exit=False)
# `exit` is deprecated, use `do_exit` instead
sys.stdout = old_stdout

PYLINT_OBJECTS = True
if hasattr(results.linter.stats, 'global_note'):
    pylint_note = results.linter.stats.global_note
    PYLINT_OBJECT_STATS = True
else:
    pylint_note = results.linter.stats['global_note']
    PYLINT_OBJECT_STATS = False


def extract_messages_by_type(type_):
    return [m for m in results.linter.reporter.messages if m.symbol == type_]


error_detected = False
error_over_ratchet_limit = False

if PYLINT_OBJECT_STATS:
    stats_by_msg = results.linter.stats.by_msg
else:
    stats_by_msg = results.linter.stats['by_msg']

for error_type, number_errors in stats_by_msg.items():
    if error_type not in UNWATCHED_ERRORS:
        if error_type in MAX_ERROR_BY_TYPE:
            max_errors = MAX_ERROR_BY_TYPE[error_type]
        else:
            max_errors = 0

        # if number_errors < max_errors - RATCHET_ERRORS:
        #     error_over_ratchet_limit = True

        if number_errors > max_errors:
            error_detected = True
            print('Fix some {} errors: {}/{}'.format(error_type,
                                                     number_errors,
                                                     max_errors))
            for message in extract_messages_by_type(error_type)[:30]:
                print('{} line {}: {}'.format(message.path, message.line, message.msg))
        elif number_errors < max_errors:
            print('You can lower number of {} to {} (actual {})'.format(
                error_type, number_errors, max_errors))


if error_detected:
    raise RuntimeError('Too many errors\nRun pylint dessia_common to get the errors')

if error_over_ratchet_limit:
    raise RuntimeError('Please lower the error limits in code_pylint.py MAX_ERROR_BY_TYPE according to warnings above')

print('Pylint note: ', pylint_note)
# if pylint_note > MIN_NOTE + RATCHET_NOTE:
#     raise ValueError(f'MIN_NOTE in code_pylint.py is too low, increase to at least {MIN_NOTE + RATCHET_NOTE}, max {pylint_note}')
if pylint_note < MIN_NOTE:
    raise ValueError(f'Pylint not is too low: {pylint_note}, expected {MIN_NOTE}')

print('You can increase MIN_NOTE in pylint to {} (actual: {})'.format(pylint_note,
                                                                      MIN_NOTE))
