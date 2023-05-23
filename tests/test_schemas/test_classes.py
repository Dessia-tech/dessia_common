from dessia_common.schemas.core import HeterogeneousSequence, DynamicDict, OptionalProperty
from typing import List, Tuple, Dict, Optional
from dessia_common.forms import StandaloneObject


schema = OptionalProperty(annotation=Optional[List[int]], attribute="optional_list", definition_default=None)
assert schema.args == (int,)
computed_schema = schema.to_dict(title="Optional List", editable=True, description="List with default value")


# TODO Should object be allowed here ?
# more_arg_tuple = schema_chunk(annotation=Tuple[int, str, StandaloneObject], title="More Arg Tuple", editable=True,
#                               description="Testing Tuple with several args")
schema = HeterogeneousSequence(annotation=Tuple[int, str], attribute="two_element_tuple")
computed_schema = schema.to_dict(title="More Arg Tuple", editable=True, description="Testing Tuple with several args")
checked_schema = schema.check_list()
assert schema.serialized == "Tuple[int, str]"
assert computed_schema == {'additionalItems': False,
                           'type': 'array',
                           'items': [{'title': 'More Arg Tuple/0', 'editable': True, 'description': '',
                                      'python_typing': 'int', 'type': 'number'},
                                     {'title': 'More Arg Tuple/1', 'editable': True, 'description': '',
                                      'python_typing': 'str', 'type': 'string'}],
                           'title': 'More Arg Tuple',
                           'editable': True,
                           'description': 'Testing Tuple with several args',
                           'python_typing': 'Tuple[int, str]'}
assert not checked_schema.checks_above_level("error")


# Dicts
schema = DynamicDict(annotation=Dict[str, int], attribute="dictionnary")
computed_schema = schema.to_dict(title="Dual Arg Dict", editable=True, description="Testing Dict with two args")
checked_schema = schema.check_list()
assert schema.serialized == "Dict[str, int]"
assert computed_schema == {'type': 'object',
                           'patternProperties': {'.*': {'type': 'number'}},
                           'title': 'Dual Arg Dict',
                           'editable': True,
                           'description': 'Testing Dict with two args',
                           'python_typing': 'Dict[str, int]'}

schema = DynamicDict(annotation=Dict[int, int], attribute="wrong_key_type_dict")
checked_schema = schema.check_list()
assert schema.serialized == "Dict[int, int]"
# computed_schema = schema.to_dict(title="Dual Arg Dict", editable=True, description="Testing Dict with two args")
assert len(checked_schema.checks_above_level("error")) == 1

schema = DynamicDict(annotation=Dict[str, Dict[str, StandaloneObject]], attribute="nested_dict")
checked_schema = schema.check_list()
assert schema.serialized == "Dict[str, Dict[str, dessia_common.forms.StandaloneObject]]"
# computed_schema = schema.to_dict(title="Deep Dict", editable=True, description="Testing Dict deeply")
assert len(checked_schema.checks_above_level("error")) == 1

print("script schemas.py has passed")
