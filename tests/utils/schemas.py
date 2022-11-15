from dessia_common.utils.schemas import schema_chunk
from typing import List
from dessia_common.core import DessiaObject

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
try:
    more_arg_list = schema_chunk(annotation=List[int, str], title="More Arg List", editable=True,
                                 description="Testing List with more args")
except TypeError:
    pass

deep_list = schema_chunk(annotation=List[List[DessiaObject]], title="Complex Deep List", editable=True,
                         description="Testin complex deep list")

