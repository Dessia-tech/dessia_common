#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from dessia_common.forms import StandaloneSubobject, StandaloneObject,\
    EmbeddedSubobject, EnhancedStandaloneSubobject,\
    InheritingStandaloneSubobject

standalone_subobject = StandaloneSubobject(floatarg=2.3)
embedded_subobject = EmbeddedSubobject(name="ES0")
dynamic_dict = {'a': True, 'b': False, 'c': False}
static_dict = {'name': 'Static Dict',
               'float_value': 93.89,
               'int_value': 10,
               'is_valid': True}

tuple_arg = ('tuple', 120)
sublist_ss = StandaloneSubobject(floatarg=-568.1)
sublist_es = EmbeddedSubobject()
enhanced_ss = EnhancedStandaloneSubobject(floatarg=-568.1, boolarg=False)
inheriting_list = [EnhancedStandaloneSubobject(floatarg=-7516.15,
                                               boolarg=True),
                   InheritingStandaloneSubobject(floatarg=1561.57,
                                                 strarg='Subclass')]

standalone_object = StandaloneObject(standalone_subobject=standalone_subobject,
                                     embedded_subobject=embedded_subobject,
                                     dynamic_dict=dynamic_dict,
                                     tuple_arg=tuple_arg,
                                     intarg=1, strarg='test_str',
                                     object_list=[sublist_ss],
                                     subobject_list=[sublist_es],
                                     builtin_list=[0, 1, 2, 3],
                                     union_arg=enhanced_ss,
                                     subclass_arg=inheriting_list[0],
                                     array_arg=[[0, 1], [1, 0]])
