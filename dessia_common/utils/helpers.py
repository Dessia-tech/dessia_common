"""
Tools for objects handling.

As a rule of thumb, functions should be placed here if:
- They can be widely used in dessia_common
- They don't have any requirements or imports

That way, we can avoid cyclic imports.
"""

import sys
from importlib import import_module
import inspect

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


def full_classname(object_, compute_for: str = 'instance'):
    """ Get full class name of object_ (module + class name). """
    if compute_for == 'instance':
        return f"{object_.__class__.__module__}.{object_.__class__.__name__}"
    if compute_for == 'class':
        return f"{object_.__module__}.{object_.__name__}"
    raise NotImplementedError(f"Cannot compute '{compute_for}' full classname for object '{object_}'")


def get_python_class_from_class_name(full_class_name: str):
    """ Get python class object corresponding to given class name. """
    cached_value = _PYTHON_CLASS_CACHE.get(full_class_name, None)
    # TODO : this is just quick fix, it will be modified soon with another.
    sys.setrecursionlimit(3000)
    if cached_value is not None:
        return cached_value

    module_name, class_name = full_class_name.rsplit('.', 1)
    module = import_module(module_name)

    class_ = getattr(module, class_name)
    # Storing in cache
    _PYTHON_CLASS_CACHE[full_class_name] = class_
    return class_


def plotdata(selector: str = None, serialize_data: bool = True, load_by_default: bool = False):
    """ Decorator to plot data."""
    def decorator(function):   
        """ Decorator to plot data. """
        function.__dict__['decorator'] = 'plotdata'
        function.__dict__['selector'] = selector
        function.__dict__['serialize_data'] = serialize_data
        function.__dict__['load_by_default'] = load_by_default
        return function
    return decorator


def markdown(selector: str = None, serialize_data: bool = False, load_by_default: bool = False):
    """ Decorator to markdown."""
    def decorator(function):   
        """ Decorator to markdown. """
        function.__dict__['decorator'] = 'markdown'
        function.__dict__['selector'] = selector
        function.__dict__['serialize_data'] = serialize_data
        function.__dict__['load_by_default'] = load_by_default
        return function
    return decorator


def get_class_and_super_class_text(class_name) -> str:
    """ Get class and super class text. """
    text_informations = []
    for class_ in inspect.getmro(class_name):
        if class_.__name__ != 'object':
            source_lines = inspect.getsourcelines(class_)
            text_informations.extend(source_lines[0])
    return text_informations
