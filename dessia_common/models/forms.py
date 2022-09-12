#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests tools
"""

from dessia_common.forms import StandaloneSubobject, EmbeddedSubobject, EnhancedStandaloneSubobject, StandaloneObject,\
    InheritingStandaloneSubobject

standalone_subobject = StandaloneSubobject(floatarg=3.78)
embedded_subobject = EmbeddedSubobject()
object_list = [StandaloneSubobject(floatarg=666.999), StandaloneSubobject(floatarg=999.666)]
subobject_list = [EmbeddedSubobject(), EmbeddedSubobject(), EmbeddedSubobject(), EmbeddedSubobject()]
union_object = EnhancedStandaloneSubobject(floatarg=333.333, boolarg=True)

subclass_arg = InheritingStandaloneSubobject(floatarg=1561.57, strarg='Subclass')

standalone_object = StandaloneObject(standalone_subobject=standalone_subobject, embedded_subobject=embedded_subobject,
                                     dynamic_dict={'key0': True, 'key1': False}, float_dict={'a': 1.3, 'b': 3.1},
                                     string_dict={'key0': 'value0', 'key1': 'value1'},
                                     tuple_arg=('Tuple', 1), intarg=5, strarg='TestStr', object_list=object_list,
                                     subobject_list=subobject_list, builtin_list=[1, 2, 3, 4, 5],
                                     union_arg=union_object, subclass_arg=subclass_arg,
                                     array_arg=[[1, 2, 3], [4, None, 6]])
