#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:24:11 2022

@author: steven
"""

from typing import TypedDict, Dict, get_type_hints, Type


class ParsedAttribute(TypedDict):
    desc: str
    type_: str
    annotation: str

class ParsedDocstring(TypedDict):
    description: str
    attributes: Dict[str, ParsedAttribute]


def parse_docstring(cls: Type) -> ParsedDocstring:
    """
    Parse docstring of given class. Refer to docs to see how docstrings
    should be built.
    """
    annotations = get_type_hints(cls.__init__)
    docstring = cls.__doc__
    if docstring:
        splitted_docstring = docstring.split(':param ')
        parsed_docstring = {"description": splitted_docstring[0].strip()}
        params = splitted_docstring[1:]
        args = {}
        for param in params:
            splitted_param = param.split(':type ')
            arg = splitted_param[0]
            typestr = splitted_param[1]
            argname, argdesc = arg.split(":", maxsplit=1)
            argtype = typestr.split(argname + ":")[-1]
            annotation = annotations[argname]
            args[argname] = {'desc': argdesc.strip(), 'type_': argtype.strip(),
                             'annotation': str(annotation)}
            # TODO Should be serialize typing ?
        parsed_docstring.update({'attributes': args})
        return parsed_docstring
    return {'description': "", 'attributes': {}}
