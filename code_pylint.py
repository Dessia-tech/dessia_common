'''
Read pylint errors to see if number of errors does not exceed specified limits
v1.0
'''

from pylint.lint import Run

MIN_NOTE = 9.05

UNWATCHED_ERRORS = ['fixme', 'trailing-whitespace', 'import-error']

MAX_ERROR_BY_TYPE = {
                     'protected-access': 28,
                     'invalid-name': 21,
                     'consider-using-f-string': 13,
                     'no-else-return': 17,
                     'arguments-differ': 5,
                     'no-member': 1,
                     'too-many-locals': 14,
                     'wrong-import-order': 11,
                     'too-many-branches': 9,
                     'unused-import': 0,
                     'unused-argument': 9,
                     'cyclic-import': 11,
                     'no-self-use': 8,
                     'unused-variable': 6,
                     'trailing-whitespace': 11,
                     'empty-docstring': 8,
                     'missing-module-docstring': 10,
                     'too-many-arguments': 6,
                     'too-few-public-methods': 5,
                     'unnecessary-comprehension': 5,
                     'no-value-for-parameter': 2,
                     'too-many-return-statements': 7,
                     'raise-missing-from': 7,
                     'consider-merging-isinstance': 6,
                     'abstract-method': 6,
                     'import-outside-toplevel': 6,
                     'too-many-instance-attributes': 4,
                     'consider-iterating-dictionary': 4,
                     'attribute-defined-outside-init': 3,
                     'simplifiable-if-expression': 3,
                     'broad-except': 3,
                     'consider-using-get': 2,
                     'undefined-loop-variable': 2,
                     'consider-using-with': 2,
                     'eval-used': 2,
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
                     'too-many-lines': 0,
                     'bare-except': 0,
                     'unspecified-encoding': 0,
                     'no-else-raise': 0,
                     'bad-indentation': 0,
                     'reimported': 0,
                     'use-implicit-booleaness-not-comparison': 0,
                     'misplaced-bare-raise': 0,
                     'redefined-argument-from-local': 0,
                     'import-error': 0,
                     'unsubscriptable-object': 0
                     }

import os
import sys
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

print('Pylint note: ', pylint_note)
assert pylint_note >= MIN_NOTE
print('You can increase MIN_NOTE in pylint to {} (actual: {})'.format(pylint_note,
                                                                      MIN_NOTE))


def extract_messages_by_type(type_):
    if PYLINT_OBJECT_STATS:
        return [m for m in results.linter.reporter.messages if m.symbol == type_]
    else:
        return [m for m in results.linter.reporter.messages if m['symbol'] == type_]


# uncontrolled_errors = {}
error_detected = False

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

        if number_errors > max_errors:
            error_detected = True
            print('Fix some {} errors: {}/{}'.format(error_type,
                                                     number_errors,
                                                     max_errors))
            for message in extract_messages_by_type(error_type):
                print('{} line {}: {}'.format(message.path, message.line, message.msg))
        elif number_errors < max_errors:
            print('You can lower number of {} to {} (actual {})'.format(
                error_type, number_errors, max_errors))


if error_detected:
    raise RuntimeError('Too many errors\nRun pylint dessia_common to get the errors')
