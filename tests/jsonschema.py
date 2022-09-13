from dessia_common.forms import StandaloneObject, StandaloneObjectWithDefaultValues, \
    ObjectWithOtherTypings, ObjectWithFaultyTyping
from dessia_common.workflow.blocks import ModelMethod, InstantiateModel
from dessia_common.models.forms import standalone_object
from dessia_common.models.workflows.workflow_from_file_input import workflow_
import dessia_common.utils.jsonschema as jss


# --- Jsonschema computation ---
jsonschema = {'definitions': {},
              '$schema': 'http://json-schema.org/draft-07/schema#',
              'type': 'object',
              'required': ['standalone_subobject', 'embedded_subobject', 'dynamic_dict', 'float_dict', 'string_dict',
                           'intarg', 'object_list', 'subobject_list', 'builtin_list', 'union_arg', 'subclass_arg',
                           'array_arg'],
              'properties': {'standalone_subobject': {'title': 'Standalone Subobject',
                                                      'editable': True,
                                                      'order': 0,
                                                      'python_typing': 'dessia_common.forms.StandaloneSubobject',
                                                      'type': 'object',
                                                      'standalone_in_db': True,
                                                      'classes': ['dessia_common.forms.StandaloneSubobject'],
                                                      'description': 'A dev subobject that is standalone_in_db'},
                             'embedded_subobject': {'title': 'Embedded Subobject',
                                                    'editable': True,
                                                    'order': 1,
                                                    'python_typing': 'dessia_common.forms.EmbeddedSubobject',
                                                    'type': 'object',
                                                    'standalone_in_db': False,
                                                    'classes': ['dessia_common.forms.EmbeddedSubobject'],
                                                    'description': "A dev subobject that isn't standalone_in_db"},
                             'dynamic_dict': {'title': 'Dynamic Dict',
                                              'editable': True,
                                              'order': 2,
                                              'python_typing': 'Dict[__builtins__.str, __builtins__.bool]',
                                              'type': 'object',
                                              'patternProperties': {'.*': {'type': 'boolean'}},
                                              'description': 'A variable length dict'},
                             'float_dict': {'title': 'Float Dict',
                                            'editable': True,
                                            'order': 3,
                                            'python_typing': 'Dict[__builtins__.str, __builtins__.float]',
                                            'type': 'object',
                                            'description': '',
                                            'patternProperties': {'.*': {'type': 'number'}}},
                             'string_dict': {'title': 'String Dict',
                                             'editable': True,
                                             'order': 4,
                                             'python_typing': 'Dict[__builtins__.str, __builtins__.str]',
                                             'type': 'object',
                                             'description': '',
                                             'patternProperties': {'.*': {'type': 'string'}}},
                             'intarg': {'title': 'Intarg',
                                        'editable': True,
                                        'order': 5,
                                        'python_typing': 'builtins.int',
                                        'description': '',
                                        'type': 'number'},

                             'object_list': {'title': 'Object List',
                                             'editable': True,
                                             'order': 6,
                                             'python_typing': 'List[dessia_common.forms.StandaloneSubobject]',
                                             'description': '',
                                             'type': 'array',
                                             'items': {'title': 'Object List',
                                                       'editable': True,
                                                       'order': 0,
                                                       'python_typing': 'dessia_common.forms.StandaloneSubobject',
                                                       'description': '',
                                                       'type': 'object',
                                                       'standalone_in_db': True,
                                                       'classes': ['dessia_common.forms.StandaloneSubobject']}},
                             'subobject_list': {'title': 'Subobject List',
                                                'editable': True,
                                                'order': 7,
                                                'python_typing': 'List[dessia_common.forms.EmbeddedSubobject]',
                                                'description': '',
                                                'type': 'array',
                                                'items': {'title': 'Subobject List',
                                                          'editable': True,
                                                          'order': 0,
                                                          'python_typing': 'dessia_common.forms.EmbeddedSubobject',
                                                          'description': '',
                                                          'type': 'object',
                                                          'standalone_in_db': False,
                                                          'classes': ['dessia_common.forms.EmbeddedSubobject']}},
                             'builtin_list': {'title': 'Builtin List',
                                              'editable': True,
                                              'order': 8,
                                              'python_typing': 'List[__builtins__.int]',
                                              'description': '',
                                              'type': 'array',
                                              'items': {'title': 'Builtin List',
                                                        'editable': True,
                                                        'order': 0,
                                                        'description': '',
                                                        'python_typing': 'builtins.int',
                                                        'type': 'number'}},
                             'union_arg': {'title': 'Union Arg',
                                           'editable': True,
                                           'order': 9,
                                           'python_typing': 'List[Union[dessia_common.forms.EmbeddedSubobject, dessia_common.forms.EnhancedEmbeddedSubobject]]',
                                           'type': 'array',
                                           'description': '',
                                           'items': {'title': 'Union Arg',
                                                     'editable': True,
                                                     'order': 0,
                                                     'description': '',
                                                     'python_typing': 'Union[dessia_common.forms.EmbeddedSubobject, dessia_common.forms.EnhancedEmbeddedSubobject]',
                                                     'type': 'object',
                                                     'classes': ['dessia_common.forms.EmbeddedSubobject',
                                                                 'dessia_common.forms.EnhancedEmbeddedSubobject'],
                                                     'standalone_in_db': False}},
                             'subclass_arg': {'title': 'Subclass Arg',
                                              'editable': True,
                                              'order': 10,
                                              'python_typing': 'InstanceOf[dessia_common.forms.StandaloneSubobject]',
                                              'type': 'object',
                                              'description': '',
                                              'instance_of': 'dessia_common.forms.StandaloneSubobject',
                                              'standalone_in_db': True},
                             'array_arg': {'title': 'Array Arg',
                                           'editable': True,
                                           'order': 11,
                                           'python_typing': 'List[List[__builtins__.float]]',
                                           'description': '',
                                           'type': 'array',
                                           'items': {'type': 'array',
                                                     'order': 0,
                                                     'python_typing': 'List[__builtins__.float]',
                                                     'items': {'title': 'Array Arg',
                                                               'editable': True,
                                                               'order': 0,
                                                               'description': '',
                                                               'python_typing': 'builtins.float',
                                                               'type': 'number'}}},
                             'name': {'title': 'Name',
                                      'editable': True,
                                      'order': 12,
                                      'description': '',
                                      'python_typing': 'builtins.str',
                                      'type': 'string',
                                      'default_value': 'Standalone Object Demo'}},
              'standalone_in_db': True,
              'description': 'Dev Object for testing purpose',
              'python_typing': "<class 'dessia_common.forms.StandaloneObject'>",
              'classes': ['dessia_common.forms.StandaloneObject'],
              'whitelist_attributes': []}

computed_jsonschema = standalone_object.jsonschema()
try:
    assert computed_jsonschema == jsonschema
except AssertionError as err:
    for key, value in computed_jsonschema['properties'].items():
        if value != jsonschema['properties'][key]:
            print('\n==', key, 'property failing ==\n')
            for subkey, subvalue in value.items():
                if subkey in jsonschema['properties'][key]:
                    check_value = jsonschema['properties'][key][subkey]
                    if subvalue != check_value:
                        print('Problematic key :', {subkey})
                        print('Computed value : ', subvalue, '\nCheck value : ', check_value)
            print('\n', value)
            print('\n', jsonschema['properties'][key])
            raise err

try:
    ObjectWithFaultyTyping.jsonschema()
except NotImplementedError:
    pass


# --- Default values ---
jsonschema = StandaloneObject.jsonschema()

assert jss.chose_default(jsonschema["properties"]["standalone_subobject"]) is None
assert jss.chose_default(jsonschema["properties"]["embedded_subobject"]) is None
assert jss.chose_default(jsonschema["properties"]["dynamic_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["float_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["string_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["intarg"]) is None
assert jss.chose_default(jsonschema["properties"]["object_list"]) is None
assert jss.chose_default(jsonschema["properties"]["subobject_list"]) is None
assert jss.chose_default(jsonschema["properties"]["builtin_list"]) is None
assert jss.chose_default(jsonschema["properties"]["union_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["subclass_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["array_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["name"]) is None

jsonschema = StandaloneObjectWithDefaultValues.jsonschema()

subobject_default_value = jss.chose_default(jsonschema["properties"]["standalone_subobject"])
assert subobject_default_value["name"] == "StandaloneSubobject1"
assert subobject_default_value["object_class"] == "dessia_common.forms.StandaloneSubobject"
assert subobject_default_value["floatarg"] == 1.7

subobject_default_value = jss.chose_default(jsonschema["properties"]["embedded_subobject"])
assert subobject_default_value["name"] == "Embedded Subobject10"
assert subobject_default_value["object_class"] == "dessia_common.forms.EmbeddedSubobject"
assert subobject_default_value["embedded_list"] == [0, 1, 2, 3, 4]

assert jss.chose_default(jsonschema["properties"]["dynamic_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["float_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["string_dict"]) is None
assert jss.chose_default(jsonschema["properties"]["intarg"]) is None  # TODO Is it ?
assert jss.chose_default(jsonschema["properties"]["object_list"]) is None
assert jss.chose_default(jsonschema["properties"]["subobject_list"]) is None
assert jss.chose_default(jsonschema["properties"]["builtin_list"]) is None
assert jss.chose_default(jsonschema["properties"]["union_arg"]) is None

subobject_default_value = jss.chose_default(jsonschema["properties"]["subclass_arg"])
assert subobject_default_value["name"] == "Inheriting Standalone Subobject1"
assert subobject_default_value["object_class"] == "dessia_common.forms.InheritingStandaloneSubobject"
assert subobject_default_value["strarg"] == "-1"
assert subobject_default_value["floatarg"] == 0.7

assert jss.chose_default(jsonschema["properties"]["array_arg"]) is None
assert jss.chose_default(jsonschema["properties"]["name"]) is None  # TODO Is it ?


# --- Datatypes ---

jsonschema = StandaloneObject.jsonschema()

assert jss.datatype_from_jsonschema(jsonschema["properties"]["standalone_subobject"]) == "standalone_object"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["embedded_subobject"]) == "embedded_object"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["dynamic_dict"]) == "dynamic_dict"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["float_dict"]) == "dynamic_dict"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["string_dict"]) == "dynamic_dict"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["intarg"]) == "builtin"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["object_list"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["subobject_list"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["builtin_list"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["union_arg"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["subclass_arg"]) == "instance_of"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["array_arg"]) == "homogeneous_sequence"
assert jss.datatype_from_jsonschema(jsonschema["properties"]["name"]) == "builtin"

jsonschema = ObjectWithOtherTypings.jsonschema()

assert jss.datatype_from_jsonschema(jsonschema["properties"]["undefined_type_attribute"]) is None

# --- Workflow Blocks ---

assert jss.datatype_from_jsonschema(ModelMethod.jsonschema()["properties"]["method_type"]) == "embedded_object"
assert jss.datatype_from_jsonschema(InstantiateModel.jsonschema()["properties"]["model_class"]) == "class"

jsonschema = workflow_._method_jsonschemas["run"]["properties"]["0"]
assert jss.datatype_from_jsonschema(jsonschema) == "file"

# --- Files ---

jsonschema = standalone_object._method_jsonschemas["generate_from_text"]["properties"]["0"]
assert jss.datatype_from_jsonschema(jsonschema) == "file"

#
# 'strarg': {'title': 'Strarg',
#                                         'editable': True,
#                                         'order': 7,
#                                         'python_typing': 'builtins.str',
#                                         'description': '',
#                                         'type': 'string'},

# 'tuple_arg': {'title': 'Tuple Arg',
#                                            'editable': True,
#                                            'order': 5,
#                                            'python_typing': 'Tuple[__builtins__.str, __builtins__.int]',
#                                            'additionalItems': False,
#                                            'type': 'array',
#                                            'items': [{'type': 'string'}, {'type': 'number'}],
#                                            'description': 'A heterogeneous sequence'},

# assert jss.chose_default(jsonschema["properties"]["tuple_arg"]) == [None, None]
# assert jss.chose_default(jsonschema["properties"]["strarg"]) is None

# assert jss.chose_default(jsonschema["properties"]["tuple_arg"]) == ("Default Tuple", 0)
# assert jss.chose_default(jsonschema["properties"]["strarg"]) is None  # TODO Is it ?
# assert jss.datatype_from_jsonschema(jsonschema["properties"]["tuple_arg"]) == "heterogeneous_sequence"
# assert jss.datatype_from_jsonschema(jsonschema["properties"]["strarg"]) == "builtin"

print("test script jsonschema.py has passed")
