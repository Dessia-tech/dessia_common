from dessia_common.schemas.core import BuiltinProperty, MeasureProperty,  ClassProperty, MethodTypeProperty,\
    HeterogeneousSequence, HomogeneousSequence, DynamicDict
from typing import List, Tuple, Dict, Type
from dessia_common.core import DessiaObject
from dessia_common.measures import Distance
from dessia_common.typings import MethodType, ClassMethodType


# Builtins
schema = BuiltinProperty(float)
computed_schema = schema.write(title="Float", editable=False, description="Testing bultins : float")
assert computed_schema == {'type': 'number', 'python_typing': 'builtins.float', 'title': 'Float',
                           'editable': False, 'description': 'Testing bultins : float'}
schema = BuiltinProperty(int)
computed_schema = schema.write(title="Integer", editable=False, description="Testing bultins : int")
assert computed_schema == {'type': 'number', 'python_typing': 'builtins.int', 'title': 'Integer',
                           'editable': False, 'description': 'Testing bultins : int'}
schema = BuiltinProperty(str)
computed_schema = schema.write(title="String", editable=False, description="Testing bultins : str")
assert computed_schema == {'type': 'string', 'python_typing': 'builtins.str', 'title': 'String',
                           'editable': False, 'description': 'Testing bultins : str'}
schema = BuiltinProperty(bool)
computed_schema = schema.write(title="Boolean", editable=False, description="Testing bultins : bool")
assert computed_schema == {'type': 'boolean', 'python_typing': 'builtins.bool', 'title': 'Boolean',
                           'editable': False, 'description': 'Testing bultins : bool'}

# Measures
schema = MeasureProperty(Distance)
computed_schema = schema.write(title="Distance", editable=True, description="Float distance")
assert computed_schema == {'type': 'number', 'python_typing': 'dessia_common.measures.Distance',
                           'title': 'Distance', 'editable': True, 'description': 'Float distance', 'si_unit': 'm'}

# Class
schema = ClassProperty(Type)
computed_schema = schema.write(title="Some class", editable=True, description="")
assert computed_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                           'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'typing.Type'}

# Methods
method_type = MethodType(class_=DessiaObject, name="to_dict")
schema = MethodTypeProperty(MethodType[DessiaObject])
computed_schema = schema.write(title="Some method", editable=True, description="to_dict")
assert computed_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                           'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'typing.Type'}

# Lists
schema = HomogeneousSequence(List)
computed_schema = schema.write(title="Bare List", editable=True, description="Testing bare List")
checked_schema = computed_schema.check("check_schema")
assert computed_schema == {'title': 'Items', 'editable': False, 'description': '',
                           'python_typing': 'typing.List', 'type': 'array', 'items': []}
assert len(checked_schema) == 1
assert checked_schema[0]["severity"] == "error"

schema = HomogeneousSequence(List[int])
computed_schema = schema.write(title="One Arg List", editable=True, description="Testing List with one arg")
checked_schema = computed_schema.check("check_schema")
assert computed_schema == {'type': 'array', 'python_typing': 'List[__builtins__.int]',
                           'items': {'type': 'number',
                                     'python_typing': 'builtins.int',
                                     'title': 'One Arg List',
                                     'editable': True,
                                     'description': ''},
                           'title': 'One Arg List', 'editable': True, 'description': 'Testing List with one arg'}
assert len(checked_schema) == 0

schema = HomogeneousSequence(List[List[DessiaObject]])
computed_schema = schema.write(title="Complex Deep List", editable=True, description="Testin complex deep list")
checked_schema = computed_schema.check("check_schema")
assert computed_schema == {'type': 'array',
                           'python_typing': 'List[List[dessia_common.core.DessiaObject]]',
                           'items': {'type': 'array',
                                     'python_typing': 'List[dessia_common.core.DessiaObject]',
                                     'items': {'type': 'object',
                                               'standalone_in_db': False,
                                               'classes': ['dessia_common.core.DessiaObject'],
                                               'title': 'Complex Deep List',
                                               'editable': True,
                                               'description': '',
                                               'python_typing': 'dessia_common.core.DessiaObject'}},
                           'title': 'Complex Deep List',
                           'editable': True,
                           'description': 'Testin complex deep list'}
assert len(checked_schema) == 0

# Tuples
# TODO Should object be allowed here ?
# more_arg_tuple = schema_chunk(annotation=Tuple[int, str, DessiaObject], title="More Arg Tuple", editable=True,
#                               description="Testing Tuple with several args")
schema = HomogeneousSequence(Tuple[int, str])
computed_schema = schema.write(title="More Arg Tuple", editable=True, description="Testing Tuple with several args")
checked_schema = computed_schema.check("check_schema")
assert computed_schema == {'additionalItems': False,
                           'type': 'array',
                           'items': [{'type': 'number'}, {'type': 'string'}],
                           'title': 'More Arg Tuple',
                           'editable': True,
                           'description': 'Testing Tuple with several args',
                           'python_typing': 'Tuple[__builtins__.int, __builtins__.str]'}
assert len(checked_schema) == 0


# Dicts
two_arg_dict = schema_chunk(annotation=Dict[str, int], title="Dual Arg Dict", editable=True,
                            description="Testing Dict with two args")
assert two_arg_dict == {'type': 'object',
                        'patternProperties': {'.*': {'type': 'number'}},
                        'title': 'Dual Arg Dict',
                        'editable': True,
                        'description': 'Testing Dict with two args',
                        'python_typing': 'Dict[__builtins__.str, __builtins__.int]'}

try:
    wrong_key_type_dict = schema_chunk(annotation=Dict[int, int], title="Dual Arg Dict", editable=True,
                                       description="Testing Dict with two args")
except NotImplementedError:
    pass

try:
    deep_dict = schema_chunk(annotation=Dict[str, Dict[str, DessiaObject]], title="Deep Dict", editable=True,
                             description="Testing Dict deeply")
except ValueError:
    # Forbidding Dict complex keys and value
    pass





print("script schemas.py has passed")
