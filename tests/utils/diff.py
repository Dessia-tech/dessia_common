from dessia_common.utils.diff import diff

value1 = {"a": 1, "b": 2, "c": 3}
value2 = {"a": 1, "b": 2}
value3 = {"a": 1}

# Checking that full path are given in result
assert diff(value1=value1, value2=value2).__repr__() == "Missing attributes:\t\t#/c: missing in second object\n"
assert diff(value1=value1, value2=value3).__repr__() == "Missing attributes:\t\t#/b: missing in second object\n\t" \
                                                        "#/c: missing in second object\n"

# Checking that diff is robust to commutation
assert diff(value1=value2, value2=value3).__repr__() == "Missing attributes:\t\t#/b: missing in second object\n"
assert diff(value1=value3, value2=value2).__repr__() == "Missing attributes:\t\t#/b: missing in first object\n"

print("script diff.py has passed")
