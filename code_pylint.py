"""
Read pylint errors to see if number of errors does not exceed specified limits
v1.2

Changes:
    v1.1: move imports to top
    v1.2: limit to 100 message to avoid overflow, global note check at end, ratchet effects
"""


import os
import sys
import random

from pylint import __version__
from pylint.lint import Run

MIN_NOTE = 8.55

UNWATCHED_ERRORS = ["fixme", "trailing-whitespace", "import-error"]

MAX_ERROR_BY_TYPE = {
    "wrong-spelling-in-docstring": 306,
    "wrong-spelling-in-comment": 87,
    "protected-access": 38,
    "consider-using-f-string": 1,
    "arguments-differ": 2,
    "no-member": 3,
    "too-many-locals": 10,  # Reduce by dropping vectored objects
    "too-many-branches": 13,
    "unused-argument": 6,
    "cyclic-import": 2,  # 0 just to test
    "no-self-use": 6,
    "trailing-whitespace": 11,
    "empty-docstring": 1,
    "missing-module-docstring": 1,
    "too-many-arguments": 21,
    "too-few-public-methods": 9,
    "unnecessary-comprehension": 1,
    "no-value-for-parameter": 2,
    "too-many-return-statements": 10,
    "consider-merging-isinstance": 1,
    "abstract-method": 6,
    "import-outside-toplevel": 4,  # TODO : will reduced in a future work (when tests are ready)
    "too-many-instance-attributes": 7,
    "no-else-raise errors": 5,
    "consider-iterating-dictionary": 1,
    "attribute-defined-outside-init": 3,
    "simplifiable-if-expression": 1,
    "broad-exception-caught": 4,
    "broad-except": 4,
    "bare-except": 4,
    "undefined-loop-variable": 1,
    "consider-using-with": 2,
    "too-many-nested-blocks": 2,
    "bad-staticmethod-argument": 1,
    "too-many-public-methods": 2,  # Try to lower by splitting DessiaObject and Workflow
    "consider-using-generator": 1,
    "too-many-statements": 2,
    "chained-comparison": 1,
    "wildcard-import": 1,
    "use-maxsplit-arg": 1,
    "duplicate-code": 1,
    "too-many-lines": 1,
}

print("pylint version: ", __version__)

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
        max_errors = MAX_ERROR_BY_TYPE.get(error_type, 0)

        if number_errors > max_errors:
            error_detected = True
            print(f"\nFix some {error_type} errors: {number_errors}/{max_errors}")

            messages = extract_messages_by_type(error_type)
            messages_to_show = sorted(random.sample(messages, min(30, len(messages))), key=lambda m: (m.path, m.line))
            for message in messages_to_show:
                print(f"{message.path} line {message.line}: {message.msg}")
        elif number_errors < max_errors:
            print(f"\nYou can lower number of {error_type} to {number_errors} (actual {max_errors})")


if error_detected:
    raise RuntimeError("Too many errors\nRun pylint dessia_common to get the errors")

if error_over_ratchet_limit:
    raise RuntimeError("Please lower the error limits in code_pylint.py MAX_ERROR_BY_TYPE according to warnings above")

print("Pylint note: ", pylint_note)
if pylint_note < MIN_NOTE:
    raise ValueError(f"Pylint not is too low: {pylint_note}, expected {MIN_NOTE}")

print("You can increase MIN_NOTE in pylint to {} (actual: {})".format(pylint_note, MIN_NOTE))
