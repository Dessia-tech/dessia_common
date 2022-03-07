#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:24:11 2022

@author: steven
"""

from typing import Dict, Any, Tuple
try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


class ParsedAttribute(TypedDict):
    desc: str
    type_: str
    annotation: str


class ParsedDocstring(TypedDict):
    description: str
    attributes: Dict[str, ParsedAttribute]


def parse_docstring(docstring: str, annotations: Dict[str, Any]) -> ParsedDocstring:
    """
    Parse docstring of given class. Refer to docs to see how docstrings
    should be built.
    """
    if docstring:
        splitted_docstring = docstring.split(':param ')
        parsed_docstring = {"description": splitted_docstring[0].strip()}
        params = splitted_docstring[1:]
        args = {}
        for param in params:
            argname, parsed_attribute = parse_attribute(param, annotations)
            args[argname] = parsed_attribute
            # TODO Should be serialize typing ?
        parsed_docstring.update({'attributes': args})
        return parsed_docstring
    return {'description': "", 'attributes': {}}


def parse_attribute(param, annotations) -> Tuple[str, ParsedAttribute]:
    splitted_param = param.split(':type ')
    arg = splitted_param[0]
    typestr = splitted_param[1]
    argname, argdesc = arg.split(":", maxsplit=1)
    argtype = typestr.split(argname + ":")[-1]
    annotation = annotations[argname]
    parsed_attribute = {'desc': argdesc.strip(),
                        'type_': argtype.strip(),
                        'annotation': str(annotation)}
    return argname, parsed_attribute


EMPTY_PARSED_ATTRIBUTE = {"desc": "", "type": "", "annotation": ""}
FAILED_DOCSTRING_PARSING = {'description': 'Docstring parsing failed', 'attributes': {}}
FAILED_ATTRIBUTE_PARSING = {"desc": 'Attribute documentation parsing failed',
                            "type": "", "annotation": ""}
