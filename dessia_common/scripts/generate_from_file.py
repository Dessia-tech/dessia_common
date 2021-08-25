from dessia_common.forms import StandaloneObject

with open('../models/data/seed_file.csv') as stream:
    obj = StandaloneObject.generate_from_file(stream)
