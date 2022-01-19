from pylint.lint import Run

MIN_NOTE = 8.3

MAX_ERROR_BY_TYPE = {'cyclic-import': 11,
                     'too-many-lines': 0,
                     'bare-except': 0,
                     'no-else-return': 27,
                     'no-self-use': 6,
                     'no-member': 18,
                     'unexpected-special-method-signature': 0,
                     'too-many-locals': 15,
                     'too-many-nested-blocks': 2,
                     'inconsistent-return-statements': 0,
                     'arguments-differ': 25,
                     'too-many-arguments': 10,
                     'undefined-variable': 0,
                     'function-redefined': 0,
                     'attribute-defined-outside-init': 3,
                     'simplifiable-if-expression': 3,
                     'redefined-builtin': 3,
                     'unnecessary-comprehension': 5,
                     'consider-using-enumerate': 0,
                     'no-value-for-parameter': 2,
                     'abstract-method': 6,
                     'wildcard-import': 1,
                     'unused-wildcard-import': 0,
                     'too-many-return-statements': 7,
                     'eval-used': 2,
                     'too-many-statements': 1,
                     'superfluous-parens': 0,
                     'chained-comparison': 1,
                     'wrong-import-order': 25,
                     'unused-variable': 12,
                     'unused-import': 43,
                     'super-init-not-called': 6,
                     'consider-using-f-string': 65,
                     'too-many-branches': 12,
                     'consider-merging-isinstance': 6,
                     'too-many-instance-attributes': 5,
                     'unused-argument': 10,
                     'undefined-loop-variable': 2,
                     'consider-using-with': 2,
                     'use-maxsplit-arg': 1,
                     'broad-except': 1,
                     'consider-iterating-dictionary': 4,
                     'raise-missing-from': 7,
                     'unspecified-encoding': 2,
                     'import-outside-toplevel': 7,
                     'consider-using-get': 2,
                     'ungrouped-imports': 3,
                     'bad-staticmethod-argument': 1,
                     'arguments-renamed': 3,
                     'too-many-public-methods': 1,
                     'consider-using-generator': 1
                     }

import os
import sys
f = open(os.devnull, 'w')

old_stdout = sys.stdout
sys.stdout = f

results = Run(['dessia_common', '--output-format=json', '--reports=no'], do_exit=False)
# `exit` is deprecated, use `do_exit` instead
sys.stdout = old_stdout

pylint_note = results.linter.stats.global_note
print('Pylint note: ', pylint_note)
assert pylint_note >= MIN_NOTE
print('You can increase MIN_NOTE in pylint to {} (actual: {})'.format(pylint_note,
                                                                      MIN_NOTE))


def extract_messages_by_type(type_):
    return [m for m in results.linter.reporter.messages if m.symbol == type_]


uncontrolled_errors = {}
error_detected = False
for error_type, number_errors in results.linter.stats.by_msg.items():
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
