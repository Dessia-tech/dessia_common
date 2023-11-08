"""
Read pylint errors to see if number of errors does not exceed specified limits
v1.3

Changes:
    v1.1: move imports to top
    v1.2: limit to 100 message to avoid overflow, global note check at end, ratchet effects
    v1.3: time decrease and simplification warning about useless entries in MAX_ERROR_BY_TYPE
"""


import os
import sys
import random
import math
from datetime import date

from pylint import __version__
from pylint.lint import Run

MIN_NOTE = 9.3

EFFECTIVE_DATE = date(2023, 1, 18)
WEEKLY_DECREASE = 0.03

UNWATCHED_ERRORS = ["fixme", "trailing-whitespace", "import-error", "protected-access"]

MAX_ERROR_BY_TYPE = {
    "protected-access": 48,  # Highly dependant on our "private" conventions. Keeps getting raised
    "arguments-differ": 1,
    "too-many-locals": 6,  # Reduce by dropping vectored objects
    "too-many-branches": 10,  # Huge refactor needed. Will be reduced by schema refactor
    "unused-argument": 6,  # Some abstract functions have unused arguments (plot_data). Hence cannot decrease
    "cyclic-import": 2,  # Still work to do on Specific based DessiaObject
    "too-many-arguments": 21,  # Huge refactor needed
    "too-few-public-methods": 3,  # Abstract classes (Errors, Checks,...)
    "too-many-return-statements": 9,  # Huge refactor needed. Will be reduced by schema refactor
    "import-outside-toplevel": 5,  # TODO : will reduced in a future work (when tests are ready)
    "too-many-instance-attributes": 7,  # Huge refactor needed (workflow, etc...)
    "broad-exception-caught": 9,  # Necessary in order not to raise non critical errors. Will be reduced by schema refactor
    "bare-except": 1,  # Necessary in order not to raise non critical errors. Will be reduced by schema refactor
    "too-many-public-methods": 2,  # Try to lower by splitting DessiaObject and Workflow
    "too-many-statements": 1,  # Will be solved by schema refactor and jsonchema removal
    "undefined-loop-variable": 1,  # Fearing to break the code by solving it
    "attribute-defined-outside-init": 3,  # For test purposes
}

ERRORS_WITHOUT_TIME_DECREASE = ['protected-access', 'arguments-differ', 'too-many-locals', 'too-many-branches',
                                'unused-argument', 'cyclic-import', 'too-many-arguments', 'too-few-public-methods',
                                'too-many-return-statements', 'import-outside-toplevel',
                                'too-many-instance-attributes', 'broad-except', 'bare-except', "broad-exception-caught",
                                'too-many-public-methods', 'too-many-statements', 'undefined-loop-variable',
                                'attribute-defined-outside-init']

print("pylint version: ", __version__)

time_decrease_coeff = 1 - (date.today() - EFFECTIVE_DATE).days / 7.0 * WEEKLY_DECREASE

f = open(os.devnull, "w")

old_stdout = sys.stdout
sys.stdout = f

results = Run(["dessia_common", "--output-format=json", "--reports=no"], do_exit=False)
# `exit` is deprecated, use `do_exit` instead
sys.stdout = old_stdout

PYLINT_OBJECTS = True
if hasattr(results.linter.stats, "global_note"):
    pylint_note = results.linter.stats.global_note
    PYLINT_OBJECT_STATS = True
else:
    pylint_note = results.linter.stats["global_note"]
    PYLINT_OBJECT_STATS = False


def extract_messages_by_type(type_):
    return [m for m in results.linter.reporter.messages if m.symbol == type_]


error_detected = False
error_over_ratchet_limit = False

if PYLINT_OBJECT_STATS:
    stats_by_msg = results.linter.stats.by_msg
else:
    stats_by_msg = results.linter.stats["by_msg"]

for error_type, number_errors in stats_by_msg.items():
    if error_type not in UNWATCHED_ERRORS:
        base_errors = MAX_ERROR_BY_TYPE.get(error_type, 0)

        if error_type in ERRORS_WITHOUT_TIME_DECREASE:
            max_errors = base_errors
        else:
            max_errors = math.ceil(base_errors * time_decrease_coeff)

        time_decrease_effect = base_errors - max_errors
        # print('time_decrease_effect', time_decrease_effect)

        if number_errors > max_errors:
            error_detected = True
            print(
                f"\nFix some {error_type} errors: {number_errors}/{max_errors} "
                f"(time effect: {time_decrease_effect} errors)")

            messages = extract_messages_by_type(error_type)
            messages_to_show = sorted(random.sample(messages, min(30, len(messages))), key=lambda m: (m.path, m.line))
            for message in messages_to_show:
                print(f"{message.path} line {message.line}: {message.msg}")
        elif number_errors < max_errors:
            print(f"\nYou can lower number of {error_type} to {number_errors+time_decrease_effect}"
                  f" (actual {base_errors})")

for error_type in MAX_ERROR_BY_TYPE:
    if error_type not in stats_by_msg:
        print(f"You can delete {error_type} entry from MAX_ERROR_BY_TYPE dict")

if error_detected:
    raise RuntimeError("Too many errors\nRun pylint dessia_common to get the errors")

if error_over_ratchet_limit:
    raise RuntimeError("Please lower the error limits in code_pylint.py MAX_ERROR_BY_TYPE according to warnings above")

print("Pylint note: ", pylint_note)
if pylint_note < MIN_NOTE:
    raise ValueError(f"Pylint not is too low: {pylint_note}, expected {MIN_NOTE}")

print("You can increase MIN_NOTE in pylint to {} (actual: {})".format(pylint_note, MIN_NOTE))
