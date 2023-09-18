import dessia_common.utils.diff as dcd
import random
from time import time

empty_sequence = []
one_element_sequence = [1]
two_element_sequence = [1, 2]

print("Building Sequences...\n")

start = time()
n_each = 200000
n_each_empty = 10000
n_each_simple = (n_each - n_each_empty)//2
same_sequences = [empty_sequence]*n_each_empty
same_sequences += [one_element_sequence]*n_each_simple
same_sequences += [two_element_sequence]*n_each_simple
same_length_sequences = [[random.randint(0, n_each) for _ in range(100)] for _ in range(n_each)]
random_length_sequences = [[random.randint(0, n_each) for _ in range(random.randint(0, 100))] for _ in range(n_each)]
sequences = same_sequences + same_length_sequences + random_length_sequences
end = time()

print(f"{len(sequences)} sequences built in {end - start}s.\n\nStarting perf testing...\n")
n_empty = sequences.count(empty_sequence) - n_each_empty
n_one = sequences.count(one_element_sequence) - n_each_simple
n_two = sequences.count(two_element_sequence) - n_each_simple
max_uniques = len(same_length_sequences) + len(random_length_sequences) - 3 + n_empty + n_one + n_two
expected = f" (Expected around {max_uniques})"


start = time()
dcd.sequence_hash(sequences)
end = time()

duration = end - start
print(f"Sequence hash : Lasted {duration}s\n")


hashes = [dcd.sequence_hash(s) for s in sequences]
unique_hashes = len(set(hashes))
total_hashes = len(hashes)
efficiency = unique_hashes/total_hashes
max_efficiency = max_uniques/total_hashes
print(f"Hash efficiency : \n"
      f" - Unique : {unique_hashes}{expected}\n"
      f" - Total : {total_hashes}\n"
      f" - Efficiency : {efficiency} (max around {max_efficiency})\n")

assert unique_hashes >= 0.90*max_uniques
assert duration <= 1e-3


def test_hash_function(hash_func):
    # Empty sequences hashes
    empty_seq1 = []
    empty_seq2 = ()
    assert hash_func(empty_seq1) == hash_func(empty_seq2)

    # Different size hashes
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3, 4]
    assert hash_func(seq1) != hash_func(seq2)

    # Different types sequences
    seq3 = [1, 2, 'a', 'b']
    seq4 = [1, 2, 'b', 'a']
    assert hash_func(seq3) != hash_func(seq4)

    # Sequences with mutables
    # TODO Add a random seed to not only check first and last, but random ones
    # TODO Not mandatory to have diffrent types
    # seq5 = [1, [2, 3], 4]
    # seq6 = [1, [3, 2], 4]
    # assert hash_func(seq5) != hash_func(seq6)

    # Sequence with different hash computation
    # TODO Check, should probably be equal
    # seq7 = [1, 2.0, 3]
    # seq8 = [1, 2, 3]
    # assert hash_func(seq7) != hash_func(seq8)

    # Random sequences
    seq11 = random.sample(range(100), 10)
    seq12 = random.sample(range(100), 10)
    assert hash_func(seq11) != hash_func(seq12)

test_hash_function(dcd.sequence_hash)


# TODO Check following for test case
# def test_hash_quality_recursive(hash_func, max_depth=5, min_size=2, max_size=10):
#     # Collision test
#     num_keys = 100000
#     collisions = 0
#     hashes = {}
#     for _ in range(num_keys):
#         key = generate_random_recursive_sequence(max_depth, min_size, max_size)
#         h = hash_func(key)
#         if h in hashes:
#             collisions += 1
#         else:
#             hashes[h] = key
#     print("Nombre de collisions:", collisions)
#
#     # Perf tests
#     import time
#     key_depths = [2, 3, 4, 5]
#     for key_depth in key_depths:
#         keys = [generate_random_recursive_sequence(key_depth, min_size, max_size) for _ in range(num_keys)]
#         start_time = time.time()
#         for key in keys:
#             # TODO Unfinished test
#
# def generate_random_recursive_sequence(max_depth, min_size, max_size):
#   if max_depth == 0:
#         return "".join(random.choices(string.ascii_letters, k=random.randint(min_size, max_size)))
#   else:
#         size = random.randint(min_size, max_size)
#         return [generate_random_recursive_sequence(max_depth - 1, min_size, max_size) for _ in
#                 range(size)]

# def test_hash_dispersion(hash_func, num_keys, num_buckets, key_length):
#     # Init a sequence that stores number of keys from each bucket
#     buckets = [0] * num_buckets
#
#     # Random sequences
#     keys = ["".join(random.choices(string.ascii_letters, k=key_length)) for _ in range(num_keys)]
#
#     # Compute hash and add to buckt
#     for key in keys:
#         bucket = hash_func(key) % num_buckets
#         buckets[bucket] += 1
#
#
#     # Compute variance and variation coeff for keys in each bucket
#     mean = num_keys / num_buckets
#     variance = sum((bucket - mean) ** 2 for bucket in buckets) / num_buckets
#     cv = (variance / mean) ** 0.5
#
#     return cv
#
# cv = test_hash_dispersion(hash_func, 100000, 1000, 20)
# print("Coefficient de variation:", cv)
print("script 'hashes.py' has passed")
