from dessia_common.utils.types import is_sequence, is_list, is_tuple, isinstance_base_types, is_simple


assert is_sequence([])
assert is_sequence((1,))
assert not is_sequence({})
assert not is_sequence(10)

assert is_list([])
assert not is_list((1,))

assert not is_tuple([])
assert is_tuple((1,))

assert not isinstance_base_types([])
assert not isinstance_base_types((1,))
assert not isinstance_base_types({})
assert isinstance_base_types(False)
assert isinstance_base_types(3.12)
assert isinstance_base_types(None)
assert isinstance_base_types(3)
assert isinstance_base_types("3")

assert not is_simple(False)
assert not is_simple(3.12)
assert is_simple(None)
assert is_simple(3)
assert is_simple("3")

print("script 'simple_types.py' has passed")
