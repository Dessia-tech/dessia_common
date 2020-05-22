from dessia_common.forms import StandaloneSubobject, StandaloneObject, EmbeddedSubobject, EnhancedStandaloneSubobject

standalone_subobject = StandaloneSubobject(floatarg=2.3)
embedded_subobject = EmbeddedSubobject(name="ES0")
dynamic_dict = {'a': True, 'b': False, 'c': False}
static_dict = {'name': 'Static Dict', 'value': 93.89, 'is_valid': True, 'subobject': EmbeddedSubobject()}
tuple_arg = ('tuple', 120)
sublist_ss = StandaloneSubobject(floatarg=-568.1)
sublist_es = EmbeddedSubobject()
enhanced_ss = EnhancedStandaloneSubobject(floatarg=568.1, boolarg=False)

standalone_object = StandaloneObject(standalone_subobject=standalone_subobject, embedded_subobject=embedded_subobject,
                                     dynamic_dict=dynamic_dict, static_dict=static_dict, tuple_arg=tuple_arg,
                                     intarg=1, strarg='test_str', object_list=[sublist_ss], subobject_list=[sublist_es],
                                     builtin_list=[0, 1, 2, 3], union_arg=enhanced_ss)
