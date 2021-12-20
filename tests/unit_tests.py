from dessia_common.forms import *
from dessia_common import enhanced_deep_attr, full_classname, DessiaObject

standalone_subobject = StandaloneSubobject(floatarg=3.78)
embedded_subobject = EmbeddedSubobject()
dynamic_dict = {'key0': True, 'key1': False}
float_dict = {'a': 1.3, 'b': 3.1}
string_dict = {'key0': 'value0', 'key1': 'value1'}
tuple_arg = ('Tuple', 1)
intarg = 5
strarg = 'TestStr'
object_list = [StandaloneSubobject(floatarg=666.999),
               StandaloneSubobject(floatarg=999.666)]
subobject_list = [EmbeddedSubobject(), EmbeddedSubobject(),
                  EmbeddedSubobject(), EmbeddedSubobject()]
builtin_list = [1, 2, 3, 4, 5]
union_object = EnhancedStandaloneSubobject(floatarg=333.333, boolarg=True)

subclass_arg = InheritingStandaloneSubobject(floatarg=1561.57,
                                             strarg='Subclass')

standalone_object = StandaloneObject(standalone_subobject=standalone_subobject,
                                     embedded_subobject=embedded_subobject,
                                     dynamic_dict=dynamic_dict,
                                     float_dict=float_dict,
                                     string_dict=string_dict,
                                     tuple_arg=tuple_arg, intarg=intarg,
                                     strarg=strarg, object_list=object_list,
                                     subobject_list=subobject_list,
                                     builtin_list=builtin_list,
                                     union_arg=union_object,
                                     subclass_arg=subclass_arg,
                                     array_arg=[[1, 2, 3], [4, None, 6]])

subobject_classname = full_classname(object_=EmbeddedSubobject,
                                     compute_for='class')
enhanced_classname = full_classname(object_=EnhancedEmbeddedSubobject,
                                    compute_for='class')

assert subobject_classname == 'dessia_common.forms.EmbeddedSubobject'
assert enhanced_classname == 'dessia_common.forms.EnhancedEmbeddedSubobject'
serialized_union = 'Union[{}, {}]'.format(subobject_classname,
                                          enhanced_classname)

# Test jsonschema
jsonschema = {'definitions': {},
 '$schema': 'http://json-schema.org/draft-07/schema#',
 'type': 'object',
 'required': ['standalone_subobject',
  'embedded_subobject',
  'dynamic_dict',
  'float_dict',
  'string_dict',
  'tuple_arg',
  'intarg',
  'strarg',
  'object_list',
  'subobject_list',
  'builtin_list',
  'union_arg',
  'subclass_arg',
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
   'patternProperties': {'.*': {'type': 'number'}}},
  'string_dict': {'title': 'String Dict',
   'editable': True,
   'order': 4,
   'python_typing': 'Dict[__builtins__.str, __builtins__.str]',
   'type': 'object',
   'patternProperties': {'.*': {'type': 'string'}}},
  'tuple_arg': {'title': 'Tuple Arg',
   'editable': True,
   'order': 5,
   'python_typing': 'Tuple[__builtins__.str, __builtins__.int]',
   'additionalItems': False,
   'type': 'array',
   'items': [{'type': 'string'}, {'type': 'number'}],
   'description': 'A heterogeneous sequence'},
  'intarg': {'title': 'Intarg',
   'editable': True,
   'order': 6,
   'python_typing': 'builtins.int',
   'type': 'number'},
  'strarg': {'title': 'Strarg',
   'editable': True,
   'order': 7,
   'python_typing': 'builtins.str',
   'type': 'string'},
  'object_list': {'title': 'Object List',
   'editable': True,
   'order': 8,
   'python_typing': 'List[dessia_common.forms.StandaloneSubobject]',
   'type': 'array',
   'items': {'title': 'Object List',
    'editable': True,
    'order': 0,
    'python_typing': 'dessia_common.forms.StandaloneSubobject',
    'type': 'object',
    'standalone_in_db': True,
    'classes': ['dessia_common.forms.StandaloneSubobject']}},
  'subobject_list': {'title': 'Subobject List',
   'editable': True,
   'order': 9,
   'python_typing': 'List[dessia_common.forms.EmbeddedSubobject]',
   'type': 'array',
   'items': {'title': 'Subobject List',
    'editable': True,
    'order': 0,
    'python_typing': 'dessia_common.forms.EmbeddedSubobject',
    'type': 'object',
    'standalone_in_db': False,
    'classes': ['dessia_common.forms.EmbeddedSubobject']}},
  'builtin_list': {'title': 'Builtin List',
   'editable': True,
   'order': 10,
   'python_typing': 'List[__builtins__.int]',
   'type': 'array',
   'items': {'title': 'Builtin List',
    'editable': True,
    'order': 0,
    'python_typing': 'builtins.int',
    'type': 'number'}},
  'union_arg': {'title': 'Union Arg',
   'editable': True,
   'order': 11,
   'python_typing': 'List[Union[dessia_common.forms.EmbeddedSubobject, dessia_common.forms.EnhancedEmbeddedSubobject]]',
   'type': 'array',
   'items': {'title': 'Union Arg',
    'editable': True,
    'order': 0,
    'python_typing': 'Union[dessia_common.forms.EmbeddedSubobject, dessia_common.forms.EnhancedEmbeddedSubobject]',
    'type': 'object',
    'classes': ['dessia_common.forms.EmbeddedSubobject',
     'dessia_common.forms.EnhancedEmbeddedSubobject'],
    'standalone_in_db': False}},
  'subclass_arg': {'title': 'Subclass Arg',
   'editable': True,
   'order': 12,
   'python_typing': 'InstanceOf[dessia_common.forms.StandaloneSubobject]',
   'type': 'object',
   'instance_of': 'dessia_common.forms.StandaloneSubobject',
   'standalone_in_db': True},
  'array_arg': {'title': 'Array Arg',
   'editable': True,
   'order': 13,
   'python_typing': 'List[List[__builtins__.float]]',
   'type': 'array',
   'items': {'type': 'array',
    'order': 0,
    'python_typing': 'List[__builtins__.float]',
    'items': {'title': 'Array Arg',
     'editable': True,
     'order': 0,
     'python_typing': 'builtins.float',
     'type': 'number'}}},
  'name': {'title': 'Name',
   'editable': True,
   'order': 14,
   'python_typing': 'builtins.str',
   'type': 'string',
   'default_value': 'Standalone Object Demo'}},
 'standalone_in_db': True,
 'description': 'Dev Object for testing purpose',
 'python_typing': "<class 'dessia_common.forms.StandaloneObject'>",
 'classes': ['dessia_common.forms.StandaloneObject'],
 'whitelist_attributes': []}

# Test deep_attr
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
                        print('Problematic key :',  {subkey})
                        print('Computed value : ', subvalue,
                              '\nCheck value : ', check_value)
            print('\n', value)
            print('\n', jsonschema['properties'][key])
            raise err

deepfloat = enhanced_deep_attr(obj=standalone_object,
                               sequence=['standalone_subobject', 'floatarg'])
assert deepfloat == 3.78

deeplist = enhanced_deep_attr(obj=standalone_object,
                              sequence=['embedded_subobject',
                                        'embedded_list', 2])
assert deeplist == 3

# deeperlist = enhanced_deep_attr(obj=standalone_object,
#                                 sequence=['subcla', 0,
#                                           'boolarg'])
# assert deeperlist

directattr = enhanced_deep_attr(obj=standalone_object, sequence=['strarg'])

assert directattr == 'TestStr'

# Test to_dict/dict_to_object
d = standalone_object.to_dict()
obj = StandaloneObject.dict_to_object(d)

assert standalone_object == obj

# Test serialization
d = standalone_object.to_dict(use_pointers=True)
assert '$ref' in d['subobject_list'][0]
o = DessiaObject.dict_to_object(d)
assert not isinstance(o.subobject_list[0], dict)

obj.to_xlsx('test')
