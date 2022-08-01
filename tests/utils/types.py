from dessia_common.utils.types import deserialize_typing, serialize_typing, MethodType, ClassMethodType, InstanceOf
from dessia_common.files import BinaryFile, StringFile
from typing import List, Tuple, Type, Dict
from dessia_common.forms import StandaloneObject

# === Typing Serialization ===
# Types
print("------------- Testing Serialization :")
test_typing = Type
serialized_typing = serialize_typing(test_typing)
assert serialized_typing == 'typing.Type'
print("ok")

print("------------- Testing deserialization : ")
deserialized_typing = deserialize_typing(serialized_typing)
assert deserialized_typing == test_typing
print("ok")