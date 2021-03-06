from dessia_common.forms import *
from dessia_common import enhanced_deep_attr

standalone_subobject = StandaloneSubobject(floatarg=3.78)
embedded_subobject = EmbeddedSubobject()
dynamic_dict = {'key0': True,
                'key1': False}
static_dict = StaticDict(name="StaticDict", float_value=-548.256,
                         int_value=10, is_valid=True)
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
                                     static_dict=static_dict,
                                     tuple_arg=tuple_arg, intarg=intarg,
                                     strarg=strarg, object_list=object_list,
                                     subobject_list=subobject_list,
                                     builtin_list=builtin_list,
                                     union_arg=union_object,
                                     subclass_arg=subclass_arg)

# Test jsonschema
jsonschema = {
    'definitions': {},
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'required': [
        'standalone_subobject', 'embedded_subobject', 'dynamic_dict',
        'static_dict', 'tuple_arg', 'intarg', 'strarg', 'object_list',
        'subobject_list', 'builtin_list', 'union_arg', 'subclass_arg'
    ],
    'properties': {
        'standalone_subobject': {
            'type': 'object', 'standalone_in_db': True,
            'title': 'Standalone Subobject', 'order': 0, 'editable': True,
            'classes': ['dessia_common.forms.StandaloneSubobject'],
            'description': 'A dev subobject that is standalone_in_db',
            'python_typing': 'dessia_common.forms.StandaloneSubobject'
        },
        'embedded_subobject': {
            'type': 'object', 'standalone_in_db': False,
            'title': 'Embedded Subobject', 'order': 1, 'editable': True,
            'classes': ['dessia_common.forms.EmbeddedSubobject'],
            'description': "A dev subobject that isn't standalone_in_db",
            'python_typing': 'dessia_common.forms.EmbeddedSubobject'
        },
        'dynamic_dict': {
            'type': 'object', 'order': 2, 'editable': True,
            'title': 'Dynamic Dict',
            'patternProperties': {'.*': {'type': 'boolean'}},
            'description': 'A variable length dict',
            'python_typing': 'Dict[__builtins__.str, __builtins__.bool]'
        },
        'static_dict': {
            'type': 'object', 'standalone_in_db': False,
            'title': 'Static Dict', 'order': 3, 'editable': True,
            'classes': ['dessia_common.forms.StaticDict'],
            'description': 'A 1-level structurewith only builtin values & str keys',
            'python_typing': 'dessia_common.forms.StaticDict'
        },
        'tuple_arg': {
            'additionalItems': False, 'type': 'array',
            'items': [{'type': 'string'}, {'type': 'number'}],
            'title': 'Tuple Arg', 'editable': True, 'order': 4,
            'description': 'A heterogeneous sequence',
            'python_typing': 'Tuple[__builtins__.str, __builtins__.int]'},
        'intarg': {
            'type': 'number', 'title': 'Intarg',
            'editable': True, 'order': 5, 'python_typing': 'builtins.int'
        },
        'strarg': {
            'type': 'string', 'title': 'Strarg',
            'editable': True, 'order': 6, 'python_typing': 'builtins.str'
        },
        'object_list': {
            'type': 'array', 'order': 7, 'editable': True,
            'title': 'Object List',
            'python_typing': 'List[dessia_common.forms.StandaloneSubobject]',
            'items': {
                'type': 'object', 'standalone_in_db': True,
                'title': 'Object List', 'order': 0, 'editable': True,
                'classes': ['dessia_common.forms.StandaloneSubobject'],
                'python_typing': 'dessia_common.forms.StandaloneSubobject'
            }
        },
        'subobject_list': {
            'type': 'array', 'order': 8, 'editable': True,
            'title': 'Subobject List',
            'python_typing': 'List[dessia_common.forms.EmbeddedSubobject]',
            'items': {
                'type': 'object', 'standalone_in_db': False,
                'title': 'Subobject List', 'order': 0, 'editable': True,
                'classes': ['dessia_common.forms.EmbeddedSubobject'],
                'python_typing': 'dessia_common.forms.EmbeddedSubobject'
            }
        },
        'builtin_list': {
            'type': 'array', 'order': 9, 'editable': True,
            'title': 'Builtin List',
            'python_typing': 'List[__builtins__.int]',
            'items': {
                'type': 'number', 'title': 'Builtin List',
                'editable': True, 'order': 0, 'python_typing': 'builtins.int'
            }
        },
        'union_arg': {
            'type': 'object', 'title': 'Union Arg',
            'classes': [
                'dessia_common.forms.StandaloneSubobject',
                'dessia_common.forms.EnhancedStandaloneSubobject'
            ],
            'python_typing': 'Union[dessia_common.forms.StandaloneSubobject, dessia_common.forms.EnhancedStandaloneSubobject]',
            'editable': True, 'order': 10, 'standalone_in_db': True
        },
        'subclass_arg': {
            'type': 'object', 'order': 11,
            'instance_of': 'dessia_common.forms.StandaloneSubobject',
            'python_typing': 'InstanceOf[dessia_common.forms.StandaloneSubobject]',
            'title': 'Subclass Arg', 'editable': True, 'standalone_in_db': True
        },
        'name': {
            'type': 'string', 'title': 'Name', 'editable': True,
            'order': 12, 'default_value': 'Standalone Object Demo',
            'python_typing': 'builtins.str'
        }
    },
    'standalone_in_db': True,
    'description': 'Dev Object for testing purpose',
    'classes': ['dessia_common.forms.StandaloneObject'],
    'whitelist_attributes': []
}

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

deepdict = enhanced_deep_attr(obj=standalone_object,
                              sequence=['static_dict', 'is_valid'])
assert deepdict is True

# deeperlist = enhanced_deep_attr(obj=standalone_object,
#                                 sequence=['subcla', 0,
#                                           'boolarg'])
# assert deeperlist

directattr = enhanced_deep_attr(obj=standalone_object, sequence=['strarg'])

assert directattr == 'TestStr'
