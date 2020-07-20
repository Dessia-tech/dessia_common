from dessia_common.forms import *
from dessia_common import get_deep_attr

standalone_subobject = StandaloneSubobject(floatarg=3.78)
embedded_subobject = EmbeddedSubobject()
dynamic_dict = {'key0': True,
                'key1': False}
static_dict = {'name': 'StaticDict',
               'value': -458.256,
               'is_valid': True,
               'subobject': EmbeddedSubobject()}
tuple_arg = ('Tuple', 1)
intarg = 5
strarg = 'TestStr'
object_list = [StandaloneSubobject(floatarg=666.999),
               StandaloneSubobject(floatarg=999.666)]
subobject_list = [EmbeddedSubobject(), EmbeddedSubobject(),
                  EmbeddedSubobject(), EmbeddedSubobject()]
builtin_list = [1, 2, 3, 4, 5]
union_object = EnhancedStandaloneSubobject(floatarg=333.333, boolarg=True)

standalone_object = StandaloneObject(standalone_subobject=standalone_subobject,
                                     embedded_subobject=embedded_subobject,
                                     dynamic_dict=dynamic_dict,
                                     static_dict=static_dict,
                                     tuple_arg=tuple_arg, intarg=intarg,
                                     strarg=strarg, object_list=object_list,
                                     subobject_list=subobject_list,
                                     builtin_list=builtin_list,
                                     union_arg=union_object)

# Test jsonschema
jsonschema = {
    'definitions': {},
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'required': ['standalone_subobject',
                 'embedded_subobject',
                 'dynamic_dict',
                 'static_dict',
                 'tuple_arg',
                 'intarg',
                 'strarg',
                 'object_list',
                 'subobject_list',
                 'builtin_list',
                 'union_arg'],
    'properties': {
        'standalone_subobject': {
            'type': 'object',
            'title': 'Standalone Subobject',
            'order': 0,
            'editable': True,
            'classes': ['dessia_common.forms.StandaloneSubobject']
        },
        'embedded_subobject': {
            'type': 'object',
            'title': 'Embedded Subobject',
            'order': 1,
            'editable': True,
            'classes': ['dessia_common.forms.EmbeddedSubobject']
        },
        'dynamic_dict': {
            'type': 'object',
            'order': 2,
            'editable': True,
            'title': 'Dynamic Dict',
            'patternProperties': {'.*': {'type': 'boolean'}}
        },
        'static_dict': {
            'definitions': {},
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'type': 'object',
            'required': ['name', 'value', 'is_valid', 'subobject'],
            'properties': {
                'name': {
                    'type': 'string',
                    'datatype': 'builtin',
                    'title': 'Name',
                    'editable': True,
                    'order': 0
                },
                'value': {
                    'type': 'number',
                    'datatype': 'builtin',
                    'title': 'Value',
                    'editable': True,
                    'order': 1
                },
                'is_valid': {
                    'type': 'boolean',
                    'datatype': 'builtin',
                    'title': 'Is Valid',
                    'editable': True,
                    'order': 2
                },
                'subobject': {
                    'type': 'object',
                    'title': 'Subobject',
                    'order': 3,
                    'editable': True,
                    'classes': ['dessia_common.forms.EmbeddedSubobject']
                }
            },
            'title': 'Static Dict',
            'order': 3,
            'editable': True,
            'classes': ['dessia_common.forms.StaticDict']
        },
        'tuple_arg': {
            'additionalItems': False,
            'type': 'array',
            'datatype': 'heterogenous_list',
            'items': [{'type': 'string'},
                      {'type': 'number'}]
        },
        'intarg': {
            'type': 'number',
            'datatype': 'builtin',
            'title': 'Intarg',
            'editable': True,
            'order': 5
        },
        'strarg': {
            'type': 'string',
            'datatype': 'builtin',
            'title': 'Strarg',
            'editable': True,
            'order': 6
        },
        'object_list': {
            'type': 'array',
            'editable': True,
            'title': 'Object List',
            'items': {
                'type': 'object',
                'title': 'Object List',
                'order': 0,
                'editable': True,
                'classes': ['dessia_common.forms.StandaloneSubobject']
            }
        },
        'subobject_list': {
            'type': 'array',
            'editable': True,
            'title': 'Subobject List',
            'items': {
                'type': 'object',
                'title': 'Subobject List',
                'order': 0,
                'editable': True,
                'classes': ['dessia_common.forms.EmbeddedSubobject']
            }
        },
        'builtin_list': {
            'type': 'array',
            'editable': True,
            'title': 'Builtin List',
            'items': {
                'type': 'number',
                'datatype': 'builtin',
                'title': 'Builtin List',
                'editable': True,
                'order': 0
            }
        },
        'union_arg': {
            'type': 'object',
            'datatype': 'union',
            'classes': ['dessia_common.forms.StandaloneSubobject',
                        'dessia_common.forms.EnhancedStandaloneSubobject'],
            'title': 'Union Arg',
            'editable': True,
            'order': 10
        },
        'name': {
            'type': 'string',
            'datatype': 'builtin',
            'title': 'Name',
            'editable': True,
            'order': 11,
            'default_value': 'Standalone Object Demo'
        }
    },
    'classes': ['dessia_common.forms.StandaloneObject'],
    'whitelist_attributes': []
}

assert standalone_object.jsonschema() == jsonschema

deepfloat = get_deep_attr(obj=standalone_object,
                          sequence=['standalone_subobject', 'floatarg'])
assert deepfloat == 3.78

deeplist = get_deep_attr(obj=standalone_object,
                         sequence=['embedded_subobject', 'embedded_list', 2])
assert deeplist == 3

deepdict = get_deep_attr(obj=standalone_object,
                         sequence=['static_dict', 'is_valid'])
assert deepdict is True

deeperlist = get_deep_attr(obj=standalone_object,
                           sequence=['static_dict', 'subobject',
                                     'embedded_list', 1])
assert deeperlist == 2

