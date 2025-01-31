"""
Tools for objects handling.

As a rule of thumb, functions should be placed here if:
- They can be widely used in dessia_common
- They don't have any requirements or imports

That way, we can avoid cyclic imports.
"""

import sys
from importlib import import_module
from ast import literal_eval
from typing import Type
from collections.abc import Sequence
from dessia_common import REF_MARKER, OLD_REF_MARKER
from dessia_common.errors import ExtractionError

_PYTHON_CLASS_CACHE = {}


def concatenate(values):
    """ Concatenate values of class class_ into a class_ containing all concatenated values. """
    types_set = set(type(value) for value in values)
    concatenated_values = None
    if len(types_set) != 1:
        raise TypeError("Block Concatenate only defined for operands of the same type.")

    first_value = values[0]
    if hasattr(first_value, 'extend'):
        concatenated_values = first_value.__class__()
        for value in values:
            concatenated_values.extend(value)

    if concatenated_values is not None:
        return concatenated_values

    raise ValueError("Block Concatenate only defined for classes with 'extend' method")


def prettyname(name: str) -> str:
    """ Create a pretty name from a string. """
    pretty_name = ''
    if name:
        strings = name.split('_')
        for i, string in enumerate(strings):
            if len(string) > 1:
                pretty_name += string[0].upper() + string[1:]
            else:
                pretty_name += string
            if i < len(strings) - 1:
                pretty_name += ' '
    return pretty_name


def full_classname(object_, compute_for: str = 'instance') -> str:
    """ Get full class name of object_ (module + class name). """
    if compute_for == 'instance':
        return f"{object_.__class__.__module__}.{object_.__class__.__name__}"
    if compute_for == 'class':
        return f"{object_.__module__}.{object_.__name__}"
    raise NotImplementedError(f"Cannot compute '{compute_for}' full classname for object '{object_}'")


def get_python_class_from_class_name(full_class_name: str) -> Type:
    """ Get python class object corresponding to given class name. """
    cached_value = _PYTHON_CLASS_CACHE.get(full_class_name, None)
    sys.setrecursionlimit(3000)  # TODO : this is just quick fix, we haven't found the real reason behind it.
    if cached_value is not None:
        return cached_value

    if "." not in full_class_name:
        return literal_eval(full_class_name)
    module_name, class_name = full_class_name.rsplit(".", 1)

    module = import_module(module_name)
    class_ = getattr(module, class_name)
    # Storing in cache
    _PYTHON_CLASS_CACHE[full_class_name] = class_
    return class_


def is_sequence(obj) -> bool:
    """
    Return True if object is sequence (but not string), else False.

    :param obj: Object to check
    :return: bool. True if object is a sequence but not a string. False otherwise
    """
    if not hasattr(obj, "__len__") or not hasattr(obj, "__getitem__"):
        # Performance improvements for trivial checks
        return False

    if is_list(obj) or is_tuple(obj):
        # Performance improvements for trivial checks
        return True
    return isinstance(obj, Sequence) and not isinstance(obj, str)


def is_list(obj) -> bool:
    """ Check if given obj is exactly of type list (not instance of). Used mainly for performance. """
    return obj.__class__ == list


def is_tuple(obj) -> bool:
    """ Check if given obj is exactly of type tuple (not instance of). Used mainly for performance. """
    return obj.__class__ == tuple


def extract_segment_from_object(object_, segment: str):
    """ Try all ways to get an attribute (segment) from an object that can of numerous types. """
    if is_sequence(object_):
        try:
            return object_[int(segment)]
        except ValueError as err:
            message_error = (f"Cannot extract segment {segment} from object {{str(object_)[:500]}}:"
                             f" segment is not a sequence index")
            raise ExtractionError(message_error) from err

    if isinstance(object_, dict):
        if segment in object_:
            return object_[segment]

        if segment.isdigit():
            intifyed_segment = int(segment)
            if intifyed_segment in object_:
                return object_[intifyed_segment]
            if segment in object_:
                return object_[segment]
            raise ExtractionError(f'Cannot extract segment {segment} from object {str(object_)[:200]}')

        # should be a tuple
        if segment.startswith('(') and segment.endswith(')') and ',' in segment:
            key = []
            for subsegment in segment.strip('()').replace(' ', '').split(','):
                if subsegment.isdigit():
                    subkey = int(subsegment)
                else:
                    subkey = subsegment
                key.append(subkey)
            return object_[tuple(key)]
        raise ExtractionError(f"Cannot extract segment {segment} from object {str(object_)[:500]}")

    # Finally, it is a regular object
    return getattr(object_, segment)


def get_in_object_from_path(object_, path, evaluate_pointers=True):
    """ Get deep attributes from an object. Argument 'path' represents path to deep attribute. """
    segments = path.lstrip('#/').split('/')
    element = object_
    for segment in segments:
        if isinstance(element, dict):
            # Going down in the object and it is a reference : evaluating sub-reference
            if evaluate_pointers:
                if REF_MARKER in element:
                    try:
                        element = get_in_object_from_path(object_, element[REF_MARKER])
                    except RecursionError as err:
                        err_msg = f'Cannot get segment {segment} from path {path} in element {str(element)[:500]}'
                        raise RecursionError(err_msg) from err
                elif OLD_REF_MARKER in element:  # Retro-compatibility to be remove sometime
                    try:
                        element = get_in_object_from_path(object_, element[OLD_REF_MARKER])
                    except RecursionError as err:
                        err_msg = f'Cannot get segment {segment} from path {path} in element {str(element)[:500]}'
                        raise RecursionError(err_msg) from err

        try:
            element = extract_segment_from_object(element, segment)
        except ExtractionError as err:

            err_msg = f'Cannot get segment {segment} from path {path} in element {str(element)[:500]}'
            raise ExtractionError(err_msg) from err

    return element


def set_in_object_from_path(object_, path, value, evaluate_pointers=True):
    """ Set deep attribute from an object to the given value. Argument 'path' represents path to deep attribute. """
    reduced_path = '/'.join(path.lstrip('#/').split('/')[:-1])
    last_segment = path.split('/')[-1]
    if reduced_path:
        last_object = get_in_object_from_path(object_, reduced_path, evaluate_pointers=evaluate_pointers)
    else:
        last_object = object_

    if is_sequence(last_object):
        last_object[int(last_segment)] = value
    elif isinstance(last_object, dict):
        last_object[last_segment] = value
    else:
        setattr(last_object, last_segment, value)
