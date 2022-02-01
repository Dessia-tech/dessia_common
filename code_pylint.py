from pylint.lint import Run

MIN_NOTE = 8.4

MAX_ERROR_BY_TYPE = {
                     'consider-using-f-string': 36,
                     'no-else-return': 23,
                     'arguments-differ': 22,
                     'no-member': 17,
                     'too-many-locals': 15,
                     'wrong-import-order': 12,
                     'too-many-branches': 12,
                     'unused-import': 10,
                     'unused-argument': 10,
                     'cyclic-import': 10,
                     'no-self-use': 7,
                     'unused-variable': 7,
                     'too-many-arguments': 10,
                     'unnecessary-comprehension': 5,
                     'no-value-for-parameter': 2,
                     'too-many-return-statements': 7,
                     'raise-missing-from': 7,
                     'consider-merging-isinstance': 6,
                     'abstract-method': 6,
                     'import-outside-toplevel': 6,
                     'too-many-instance-attributes': 5,
                     'consider-iterating-dictionary': 4,
                     'attribute-defined-outside-init': 3,
                     'simplifiable-if-expression': 3,
                     'redefined-builtin': 3,
                     'broad-except': 3,
                     'unspecified-encoding': 2,
                     'consider-using-get': 2,
                     'undefined-loop-variable': 2,
                     'consider-using-with': 2,
                     'eval-used': 2,
                     'too-many-nested-blocks': 2,
                     'bad-staticmethod-argument': 1,
                     'too-many-public-methods': 1,
                     'consider-using-generator': 1,
                     'too-many-statements': 1,
                     'chained-comparison': 1,
                     'wildcard-import': 1,
                     'use-maxsplit-arg': 1,
                     # No tolerance errors
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
if hasattr(results.linter.stats,'global_note'):
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
    return [m for m in results.linter.reporter.messages if m.symbol == type_]


uncontrolled_errors = {}
error_detected = False

if PYLINT_OBJECT_STATS:
    stats_by_msg = results.linter.stats.by_msg
else:
    stats_by_msg = results.linter.stats['by_msg']

for error_type, number_errors in stats_by_msg.items():
    if error_type in MAX_ERROR_BY_TYPE:
        if number_errors > MAX_ERROR_BY_TYPE[error_type]:
            error_detected = True
            print('Fix some {} errors: {}/{}'.format(error_type,
                                                     number_errors,
                                                     MAX_ERROR_BY_TYPE[error_type]))
            for message in extract_messages_by_type(error_type):
                print('{} line {}: {}'.format(message.path, message.line, message.msg))
        elif number_errors < MAX_ERROR_BY_TYPE[error_type]:
            print('You can lower number of {} to {} (actual {})'.format(
                error_type, number_errors, MAX_ERROR_BY_TYPE[error_type]))

    else:
        if not error_type in uncontrolled_errors:
            uncontrolled_errors[error_type] = number_errors

if uncontrolled_errors:
    print('Uncontrolled errors', uncontrolled_errors)

if error_detected:
    raise RuntimeError('Too many errors\nRun pylint dessia_common to get the errors')
