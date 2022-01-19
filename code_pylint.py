from pylint.lint import Run

MIN_NOTE = 8.65

MAX_ERROR_BY_TYPE = {'cyclic-import': 8,
                     'too-many-lines': 0,
                     'bare-except': 0,
                     'no-else-return': 25,
                     'no-self-use': 6,
                     'no-member': 31,
                     'unexpected-special-method-signature': 0,
                     'too-many-locals': 15,
                     'too-many-nested-blocks': 2,
                     'inconsistent-return-statements': 0,
                     'arguments-differ': 28,
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
                     'chained-comparison': 1
                     }

results = Run(['dessia_common', '--output-format=json'], do_exit=False)
# `exit` is deprecated, use `do_exit` instead
print(results.linter.stats['global_note'])

assert results.linter.stats['global_note'] >= MIN_NOTE
print('You can increase MIN_NOTE in pylint to {} (actual: {})'.format(results.linter.stats['global_note'],
                                                                      MIN_NOTE))


def extract_messages_by_type(type_):
    return [m for m in results.linter.reporter.messages if m['symbol'] == type_]


uncontrolled_errors = {}
error_detected = False
for error_type, number_errors in results.linter.stats['by_msg'].items():
    if error_type in MAX_ERROR_BY_TYPE:
        if number_errors > MAX_ERROR_BY_TYPE[error_type]:
            error_detected = True
            print('Fix some {} errors: {}/{}'.format(error_type,
                                                     number_errors,
                                                     MAX_ERROR_BY_TYPE[error_type]))
            for message in extract_messages_by_type(error_type):
                print('{} line {}: {}'.format(message['path'], message['line'], message['message']))
        elif number_errors < MAX_ERROR_BY_TYPE[error_type]:
            print('You can lower number of {} to {} (actual {})'.format(
                error_type, number_errors, MAX_ERROR_BY_TYPE[error_type]))

    else:
        if not error_type in uncontrolled_errors:
            uncontrolled_errors[error_type] = number_errors

print('Uncontrolled errors', uncontrolled_errors)

if error_detected:
    raise RuntimeError('Too many errors\nRun pylint dessia_common to get the errors'.format(
        error_type, number_errors))
