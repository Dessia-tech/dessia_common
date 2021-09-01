from dessia_common.forms import StandaloneObject
import io

with open('../models/data/seed_file.csv') as stream:
    obj = StandaloneObject.generate_from_text(stream)

with open('../models/data/seed_file.csv') as stream:
    string = stream.read()
    bytes_content = string.encode('utf-8')
binbuffer = io.BytesIO(bytes_content)
strbuffer = io.StringIO(string)

StandaloneObject.generate_from_text(strbuffer)
StandaloneObject.generate_from_bin(binbuffer)

argument_dict = {'0': string}

deserialized_dict = obj.dict_to_arguments(argument_dict, 'generate_from_text')

