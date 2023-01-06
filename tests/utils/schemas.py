from dessia_common.schemas.core import BuiltinProperty, MeasureProperty,  ClassProperty, MethodTypeProperty,\
    HeterogeneousSequence, HomogeneousSequence, DynamicDict
from typing import List, Tuple, Dict, Type
from dessia_common.forms import StandaloneObject
from dessia_common.measures import Distance
from dessia_common.typings import MethodType


# Builtins
schema = BuiltinProperty(float)
computed_schema = schema.to_dict(title="Float", editable=False, description="Testing bultins : float")
assert computed_schema == {'type': 'number', 'python_typing': 'builtins.float', 'title': 'Float',
                           'editable': False, 'description': 'Testing bultins : float'}
schema = BuiltinProperty(int)
computed_schema = schema.to_dict(title="Integer", editable=False, description="Testing bultins : int")
assert computed_schema == {'type': 'number', 'python_typing': 'builtins.int', 'title': 'Integer',
                           'editable': False, 'description': 'Testing bultins : int'}
schema = BuiltinProperty(str)
computed_schema = schema.to_dict(title="String", editable=False, description="Testing bultins : str")
assert computed_schema == {'type': 'string', 'python_typing': 'builtins.str', 'title': 'String',
                           'editable': False, 'description': 'Testing bultins : str'}
schema = BuiltinProperty(bool)
computed_schema = schema.to_dict(title="Boolean", editable=False, description="Testing bultins : bool")
assert computed_schema == {'type': 'boolean', 'python_typing': 'builtins.bool', 'title': 'Boolean',
                           'editable': False, 'description': 'Testing bultins : bool'}

# Measures
schema = MeasureProperty(Distance)
computed_schema = schema.to_dict(title="Distance", editable=True, description="Float distance")
assert computed_schema == {'type': 'number', 'python_typing': 'dessia_common.measures.Distance',
                           'title': 'Distance', 'editable': True, 'description': 'Float distance', 'si_unit': 'm'}

# Class
schema = ClassProperty(Type)
computed_schema = schema.to_dict(title="Some class", editable=True, description="")
assert computed_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                           'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'typing.Type'}

# Methods
method_type = MethodType(class_=StandaloneObject, name="to_dict")
schema = MethodTypeProperty(MethodType[StandaloneObject])
computed_schema = schema.to_dict(title="Some method", editable=True, description="to_dict")
assert computed_schema == {'type': 'object', 'is_method': True, 'title': 'Some method', 'editable': True,
                           'description': "to_dict", "classmethod_": False,
                           'python_typing': "MethodType[dessia_common.core.StandaloneObject]",
                           'properties': {
                               'name': {'type': 'string'},
                               'class_': {
                                   'title': 'Some method',
                                   'editable': True,
                                   'description': 'to_dict',
                                   'python_typing': 'dessia_common.core.StandaloneObject',
                                   'type': 'object',
                                   'standalone_in_db': False,
                                   'classes': ['dessia_common.core.StandaloneObject']
                               }
                           }}

# Lists
schema = HomogeneousSequence(List)
computed_schema = schema.to_dict(title="Bare List", editable=True, description="Testing bare List")
checked_schema = schema.check_list("check_schema")
assert computed_schema == {'title': 'Bare List', 'editable': True, 'description': "Testing bare List",
                           'python_typing': 'typing.List', 'type': 'array', 'items': []}
assert checked_schema.checks_above_level("error")
assert len(checked_schema) == 1
assert checked_schema[0].level == "error"

schema = HomogeneousSequence(List[int])
computed_schema = schema.to_dict(title="One Arg List", editable=True, description="Testing List with one arg")
checked_schema = schema.check_list("check_schema")
assert computed_schema == {'type': 'array', 'python_typing': 'List[__builtins__.int]',
                           'items': {'type': 'number',
                                     'python_typing': 'builtins.int',
                                     'title': 'One Arg List',
                                     'editable': True,
                                     'description': ''},
                           'title': 'One Arg List', 'editable': True, 'description': 'Testing List with one arg'}
assert not checked_schema.checks_above_level("error")

schema = HomogeneousSequence(List[List[StandaloneObject]])
computed_schema = schema.to_dict(title="Complex Deep List", editable=True, description="Testin complex deep list")
checked_schema = schema.check_list("check_schema")
assert computed_schema == {'type': 'array',
                           'python_typing': 'List[List[dessia_common.core.StandaloneObject]]',
                           'items': {'type': 'array',
                                     'python_typing': 'List[dessia_common.core.StandaloneObject]',
                                     'items': {'type': 'object',
                                               'standalone_in_db': False,
                                               'classes': ['dessia_common.core.StandaloneObject'],
                                               'title': 'Complex Deep List',
                                               'editable': True,
                                               'description': '',
                                               'python_typing': 'dessia_common.core.StandaloneObject'}},
                           'title': 'Complex Deep List',
                           'editable': True,
                           'description': 'Testin complex deep list'}
assert not checked_schema.checks_above_level("error")

# Tuples
schema = HeterogeneousSequence(Tuple)
computed_schema = schema.to_dict(title="Bare Tuple", editable=True, description="Testing Tuple with no arg")
checked_schema = schema.check_list("check_schema")
assert computed_schema == {'additionalItems': False,
                           'type': 'array',
                           'items': [],
                           'title': 'Bare Tuple',
                           'editable': True,
                           'description': 'Testing Tuple with no arg',
                           'python_typing': 'Tuple'}
assert checked_schema.checks_above_level("error")
assert len(checked_schema) == 1
assert checked_schema[0].level == "error"

# TODO Should object be allowed here ?
# more_arg_tuple = schema_chunk(annotation=Tuple[int, str, StandaloneObject], title="More Arg Tuple", editable=True,
#                               description="Testing Tuple with several args")
schema = HeterogeneousSequence(Tuple[int, str])
computed_schema = schema.to_dict(title="More Arg Tuple", editable=True, description="Testing Tuple with several args")
checked_schema = schema.check_list("check_schema")
assert computed_schema == {'additionalItems': False,
                           'type': 'array',
                           'items': [{'type': 'number'}, {'type': 'string'}],
                           'title': 'More Arg Tuple',
                           'editable': True,
                           'description': 'Testing Tuple with several args',
                           'python_typing': 'Tuple[__builtins__.int, __builtins__.str]'}
assert not checked_schema.checks_above_level("error")


# Dicts
schema = DynamicDict(Dict[str, int])
computed_schema = schema.to_dict(title="Dual Arg Dict", editable=True, description="Testing Dict with two args")
checked_schema = schema.check_list("check_schema")
assert computed_schema == {'type': 'object',
                           'patternProperties': {'.*': {'type': 'number'}},
                           'title': 'Dual Arg Dict',
                           'editable': True,
                           'description': 'Testing Dict with two args',
                           'python_typing': 'Dict[__builtins__.str, __builtins__.int]'}

schema = DynamicDict(Dict[int, int])
checked_schema = schema.check_list("check_schema")
# computed_schema = schema.to_dict(title="Dual Arg Dict", editable=True, description="Testing Dict with two args")
assert len(checked_schema.checks_above_level("error")) == 1

schema = DynamicDict(Dict[str, Dict[str, StandaloneObject]])
checked_schema = schema.check_list("check_schema")
computed_schema = schema.to_dict(title="Deep Dict", editable=True, description="Testing Dict deeply")
assert len(checked_schema.checks_above_level("error")) == 1

print("script schemas.py has passed")
