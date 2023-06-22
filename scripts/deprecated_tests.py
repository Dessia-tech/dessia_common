from dessia_common.utils.jsonschema import default_dict
from dessia_common.utils.serialization import serialize, serialize_with_pointers, deserialize, dict_to_object
from dessia_common.forms import StandaloneObject

obj = StandaloneObject.generate(1)
schema = obj.schema()

defdict = default_dict(schema)
serialized = serialize(obj)
# serialized_with_pointers = serialize_with_pointers(obj)
deserialized = deserialize(serialized)
new_obj = dict_to_object(serialized)

print("Old tests passed.")
