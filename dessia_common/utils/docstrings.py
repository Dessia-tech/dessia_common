#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module for docstring parsing to platform and Sphinx auto documentation. """

from inspect import isclass, ismethod, isfunction
from typing import Dict, Any, Tuple, get_type_hints
from dessia_common.utils.types import serialize_typing

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


class ParsedAttribute(TypedDict):
    """ Parsed description of a docstring attribute. """

    desc: str
    type_: str
    annotation: str


class ParsedDocstring(TypedDict):
    """ Parsed description of a docstring. """

    description: str
    attributes: Dict[str, ParsedAttribute]


def parse_class_docstring(class_) -> ParsedDocstring:
    """ Helper to get parse docstring from a class. """
    docstring = class_.__doc__
    annotations = get_type_hints(class_.__init__)
    return parse_docstring(docstring=docstring, annotations=annotations)


def parse_docstring(docstring: str, annotations: Dict[str, Any]) -> ParsedDocstring:
    """ Parse user-defined docstring of given class. Refer to docs to see how docstrings should be built. """
    if docstring:
        no_return_docstring = docstring.split(':return:')[0]
        splitted_docstring = no_return_docstring.split(':param ')
        parsed_docstring = {"description": splitted_docstring[0].strip()}
        params = splitted_docstring[1:]
        args = {}
        for param in params:
            argname, parsed_attribute = parse_attribute(param, annotations)
            args[argname] = parsed_attribute
        parsed_docstring.update({'attributes': args})
        return parsed_docstring
    return {'description': "", 'attributes': {}}


def parse_attribute(param, annotations) -> Tuple[str, ParsedAttribute]:
    """ Extract attribute from user-defined docstring. """
    if ":type" in param:
        param = param.split(":type ")[0]
    argname, argdesc = param.split(":", maxsplit=1)
    annotation = annotations[argname]
    parsed_attribute = {'desc': argdesc.strip(), 'type_': serialize_typing(annotation), 'annotation': str(annotation)}
    return argname, parsed_attribute


EMPTY_PARSED_ATTRIBUTE = {"desc": "", "type": "", "annotation": ""}
FAILED_DOCSTRING_PARSING = {'description': 'Docstring parsing failed', 'attributes': {}}
FAILED_ATTRIBUTE_PARSING = {"desc": 'Attribute documentation parsing failed',
                            "type": "", "annotation": ""}


def _check_docstring(element):
    """ Return True if an object, a class or a method have a proper docstring. Otherwise, return False. """
    docstring = element.__doc__
    if docstring is None:
        print(f'Docstring not found for {element}')
        return False
    if isclass(element):
        # element is an object or a class
        annotations = get_type_hints(element.__init__)
    elif ismethod(element) or isfunction(element):
        # element is a method
        annotations = get_type_hints(element)
    else:
        raise NotImplementedError
    try:
        parse_docstring(docstring=docstring,
                        annotations=annotations)
        return True
    except TypeError:
        return False
