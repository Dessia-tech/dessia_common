from dessia_common.schemas.core import BuiltinProperty, MeasureProperty,  ClassProperty, MethodTypeProperty,\
    HeterogeneousSequence, HomogeneousSequence, DynamicDict
from typing import List, Tuple, Dict, Type
from dessia_common.forms import StandaloneObject
from dessia_common.measures import Distance
from dessia_common.typings import MethodType


# Builtins
schema = BuiltinProperty(annotation=float, attribute="floating")
computed_schema = schema.to_dict(title="Float", editable=False, description="Testing bultins : float")
assert schema.serialized == "float"
assert computed_schema == {'type': 'number', 'python_typing': 'float', 'title': 'Float',
                           'editable': False, 'description': 'Testing bultins : float'}
schema = BuiltinProperty(annotation=int, attribute="integer")
computed_schema = schema.to_dict(title="Integer", editable=False, description="Testing bultins : int")
assert schema.serialized == "int"
assert computed_schema == {'type': 'number', 'python_typing': 'int', 'title': 'Integer',
                           'editable': False, 'description': 'Testing bultins : int'}
schema = BuiltinProperty(annotation=str, attribute="string")
computed_schema = schema.to_dict(title="String", editable=False, description="Testing bultins : str")
assert schema.serialized == "str"
assert computed_schema == {'type': 'string', 'python_typing': 'str', 'title': 'String',
                           'editable': False, 'description': 'Testing bultins : str'}
schema = BuiltinProperty(annotation=bool, attribute="boolean")
computed_schema = schema.to_dict(title="Boolean", editable=False, description="Testing bultins : bool")
assert schema.serialized == "bool"
assert computed_schema == {'type': 'boolean', 'python_typing': 'bool', 'title': 'Boolean',
                           'editable': False, 'description': 'Testing bultins : bool'}

# Measures
schema = MeasureProperty(annotation=Distance, attribute="distance")
computed_schema = schema.to_dict(title="Distance", editable=True, description="Float distance")
assert schema.serialized == "dessia_common.measures.Distance"
assert computed_schema == {'type': 'number', 'python_typing': 'dessia_common.measures.Distance',
                           'title': 'Distance', 'editable': True, 'description': 'Float distance', 'si_unit': 'm'}

# Class
schema = ClassProperty(annotation=Type, attribute="custom_class")
assert schema.serialized == "type"
computed_schema = schema.to_dict(title="Some class", editable=True, description="")
assert computed_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                           'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'type'}

schema = ClassProperty(annotation=Type[StandaloneObject], attribute="custom_class")
computed_schema = schema.to_dict(title="Some class", editable=True, description="")
assert schema.serialized == "type[dessia_common.forms.StandaloneObject]"
assert computed_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                           'title': 'Some class', 'editable': True, 'description': '',
                           'python_typing': 'type[dessia_common.forms.StandaloneObject]'}

# Methods
method_type = MethodType(class_=StandaloneObject, name="to_dict")
schema = MethodTypeProperty(annotation=MethodType[StandaloneObject], attribute="method")
computed_schema = schema.to_dict(title="Some method", editable=True, description="to_dict")
assert schema.serialized == "MethodType[dessia_common.forms.StandaloneObject]"
assert computed_schema == {'type': 'object', 'is_method': True, 'title': 'Some method', 'editable': True,
                           'description': "to_dict", "classmethod_": False,
                           'python_typing': "MethodType[dessia_common.forms.StandaloneObject]",
                           'properties': {
                               'name': {'type': 'string'},
                               'class_': {
                                   'title': 'Some method',
                                   'editable': True,
                                   'description': 'to_dict',
                                   'python_typing': 'dessia_common.forms.StandaloneObject',
                                   'type': 'object',
                                   'standalone_in_db': True,
                                   'classes': ['dessia_common.forms.StandaloneObject']
                               }
                           }}

# Lists
schema = HomogeneousSequence(annotation=List, attribute="bare_list")
checked_schema = schema.check_list()
assert schema.serialized == "list"
assert checked_schema.checks_above_level("error")
assert len(checked_schema) == 2
assert checked_schema[0].level == "error"

schema = HomogeneousSequence(annotation=List[int], attribute="integer_list")
computed_schema = schema.to_dict(title="One Arg List", editable=True, description="Testing List with one arg")
checked_schema = schema.check_list()
assert schema.serialized == "list[int]"
assert computed_schema == {'type': 'array', 'python_typing': 'list[int]',
                           'items': {'type': 'number',
                                     'python_typing': 'int',
                                     'title': 'One Arg List/0',
                                     'editable': True,
                                     'description': ''},
                           'title': 'One Arg List', 'editable': True, 'description': 'Testing List with one arg'}
assert not checked_schema.checks_above_level("error")

schema = HomogeneousSequence(annotation=List[List[StandaloneObject]], attribute="nested_list")
computed_schema = schema.to_dict(title="Complex Deep List", editable=True, description="Testing complex deep list")
checked_schema = schema.check_list()
assert schema.serialized == "list[list[dessia_common.forms.StandaloneObject]]"
assert computed_schema == {'type': 'array',
                           'python_typing': 'list[list[dessia_common.forms.StandaloneObject]]',
                           'items': {'type': 'array',
                                     'title': 'Complex Deep List/0',
                                     'editable': True,
                                     'description': '',
                                     'python_typing': 'list[dessia_common.forms.StandaloneObject]',
                                     'items': {'type': 'object',
                                               'standalone_in_db': True,
                                               'classes': ['dessia_common.forms.StandaloneObject'],
                                               'title': 'Complex Deep List/0/0',
                                               'editable': True,
                                               'description': '',
                                               'python_typing': 'dessia_common.forms.StandaloneObject'}},
                           'title': 'Complex Deep List',
                           'editable': True,
                           'description': 'Testing complex deep list'}
assert not checked_schema.checks_above_level("error")

# Tuples
schema = HeterogeneousSequence(annotation=Tuple, attribute="bare_tuple")
computed_schema = schema.to_dict(title="Bare Tuple", editable=True, description="Testing Tuple with no arg")
checked_schema = schema.check_list()
assert schema.serialized == "tuple"
assert computed_schema == {'additionalItems': False,
                           'type': 'array',
                           'items': [],
                           'title': 'Bare Tuple',
                           'editable': True,
                           'description': 'Testing Tuple with no arg',
                           'python_typing': 'tuple'}
assert checked_schema.checks_above_level("error")
assert len(checked_schema) == 1
assert checked_schema[0].level == "error"

# TODO Should object be allowed here ?
# more_arg_tuple = schema_chunk(annotation=Tuple[int, str, StandaloneObject], title="More Arg Tuple", editable=True,
#                               description="Testing Tuple with several args")
schema = HeterogeneousSequence(annotation=Tuple[int, str], attribute="two_element_tuple")
computed_schema = schema.to_dict(title="More Arg Tuple", editable=True, description="Testing Tuple with several args")
checked_schema = schema.check_list()
assert schema.serialized == "tuple[int, str]"
assert computed_schema == {'additionalItems': False,
                           'type': 'array',
                           'items': [{'title': 'More Arg Tuple/0', 'editable': True, 'description': '',
                                      'python_typing': 'int', 'type': 'number'},
                                     {'title': 'More Arg Tuple/1', 'editable': True, 'description': '',
                                      'python_typing': 'str', 'type': 'string'}],
                           'title': 'More Arg Tuple',
                           'editable': True,
                           'description': 'Testing Tuple with several args',
                           'python_typing': 'tuple[int, str]'}
assert not checked_schema.checks_above_level("error")


# Dicts
schema = DynamicDict(annotation=Dict[str, int], attribute="dictionnary")
computed_schema = schema.to_dict(title="Dual Arg Dict", editable=True, description="Testing Dict with two args")
checked_schema = schema.check_list()
assert schema.serialized == "dict[str, int]"
assert computed_schema == {'type': 'object',
                           'patternProperties': {'.*': {'type': 'number'}},
                           'title': 'Dual Arg Dict',
                           'editable': True,
                           'description': 'Testing Dict with two args',
                           'python_typing': 'dict[str, int]'}

schema = DynamicDict(annotation=Dict[int, int], attribute="wrong_key_type_dict")
checked_schema = schema.check_list()
assert schema.serialized == "dict[int, int]"
# computed_schema = schema.to_dict(title="Dual Arg Dict", editable=True, description="Testing Dict with two args")
assert len(checked_schema.checks_above_level("error")) == 1

schema = DynamicDict(annotation=Dict[str, Dict[str, StandaloneObject]], attribute="nested_dict")
checked_schema = schema.check_list()
assert schema.serialized == "dict[str, dict[str, dessia_common.forms.StandaloneObject]]"
# computed_schema = schema.to_dict(title="Deep Dict", editable=True, description="Testing Dict deeply")
assert len(checked_schema.checks_above_level("error")) == 1

print("script schemas.py has passed")
