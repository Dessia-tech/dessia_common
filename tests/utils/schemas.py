from dessia_common.utils.schemas import schema_chunk
from typing import List, Tuple, Dict, Type
from dessia_common.core import DessiaObject
from dessia_common.measures import Distance
from dessia_common.typings import MethodType, ClassMethodType


# Builtins
float_schema = schema_chunk(annotation=float, title="Float", editable=False, description="Testing bultins : float")
assert float_schema == {'type': 'number', 'python_typing': 'builtins.float', 'title': 'Float',
                        'editable': False, 'description': 'Testing bultins : float'}
int_schema = schema_chunk(annotation=int, title="Integer", editable=False, description="Testing bultins : int")
assert int_schema == {'type': 'number', 'python_typing': 'builtins.int', 'title': 'Integer',
                      'editable': False, 'description': 'Testing bultins : int'}
str_schema = schema_chunk(annotation=str, title="String", editable=False, description="Testing bultins : str")
assert str_schema == {'type': 'string', 'python_typing': 'builtins.str', 'title': 'String',
                      'editable': False, 'description': 'Testing bultins : str'}
bool_schema = schema_chunk(annotation=bool, title="Boolean", editable=False, description="Testing bultins : bool")
assert bool_schema == {'type': 'boolean', 'python_typing': 'builtins.bool', 'title': 'Boolean',
                       'editable': False, 'description': 'Testing bultins : bool'}

# Measures
distance_schema = schema_chunk(annotation=Distance, title="Distance", editable=True, description="Float distance")
assert distance_schema == {'type': 'number', 'python_typing': 'dessia_common.measures.Distance', 'title': 'Distance',
                           'editable': True, 'description': 'Float distance', 'si_unit': 'm'}

# Class
class_schema = schema_chunk(annotation=Type, title="Some class", editable=True, description="")
assert class_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                        'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'typing.Type'}

# Methods
method_type = MethodType(class_=DessiaObject, name="to_dict")
method_schema = schema_chunk(annotation=method_type, title="Some method", editable=True, description="to_dict")
assert class_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                        'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'typing.Type'}

class_schema = schema_chunk(annotation=Type, title="Some class", editable=True, description="")
assert class_schema == {'type': 'object', 'is_class': True, 'properties': {'name': {'type': 'string'}},
                        'title': 'Some class', 'editable': True, 'description': '', 'python_typing': 'typing.Type'}

# Lists
try:
    bare_list = schema_chunk(annotation=List, title="Bare List", editable=True, description="Testing bare List")
except NotImplementedError:
    pass

one_arg_list = schema_chunk(annotation=List[int], title="One Arg List", editable=True,
                            description="Testing List with one arg")
assert one_arg_list == {'type': 'array', 'python_typing': 'List[__builtins__.int]',
                        'items': {'type': 'number',
                                  'python_typing': 'builtins.int',
                                  'title': 'One Arg List',
                                  'editable': True,
                                  'description': ''},
                        'title': 'One Arg List', 'editable': True, 'description': 'Testing List with one arg'}

deep_list = schema_chunk(annotation=List[List[DessiaObject]], title="Complex Deep List", editable=True,
                         description="Testin complex deep list")
assert deep_list == {'type': 'array',
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

# Tuples
# TODO Should object be allowed here ?
# more_arg_tuple = schema_chunk(annotation=Tuple[int, str, DessiaObject], title="More Arg Tuple", editable=True,
#                               description="Testing Tuple with several args")
more_arg_tuple = schema_chunk(annotation=Tuple[int, str], title="More Arg Tuple", editable=True,
                              description="Testing Tuple with several args")
assert more_arg_tuple == {'additionalItems': False,
                          'type': 'array',
                          'items': [{'type': 'number'}, {'type': 'string'}],
                          'title': 'More Arg Tuple',
                          'editable': True,
                          'description': 'Testing Tuple with several args',
                          'python_typing': 'Tuple[__builtins__.int, __builtins__.str]'}


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
