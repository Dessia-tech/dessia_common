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

deepfloat = get_deep_attr(obj=standalone_object,
                          sequence=['standalone_subobject', 'floatarg'])
print(deepfloat)

deeplist = get_deep_attr(obj=standalone_object,
                         sequence=['embedded_subobject', 'embedded_list', 2])
print(deeplist)

deepdict = get_deep_attr(obj=standalone_object,
                         sequence=['static_dict', 'is_valid'])
print(deepdict)

deeperlist = get_deep_attr(obj=standalone_object,
                           sequence=['static_dict', 'subobject',
                                     'embedded_list', 1])
print(deeperlist)
