""" Provides decorators that work as 'flags' for display settings. """

from typing import Type, List
import inspect
import ast


DISPLAY_DECORATORS = ["plot_data_display", "markdown_display", "cad_display"]


def get_all_decorated_methods(class_: Type) -> List[ast.FunctionDef]:
    """ Get all decorated method from class_. """
    class_source = inspect.getsource(class_)
    ast_tree = ast.parse(source=class_source)
    class_body = ast_tree.body[0]
    return [m for m in class_body.body if isinstance(m, ast.FunctionDef) and m.decorator_list]


def get_decorated_methods(class_: Type, decorator_name: str):
    """ Get all method from class_ that are decorated with decorator_name. """
    methods = get_all_decorated_methods(class_)
    method_names = []

    for method in methods:
        decorator_names = []
        for decorator in method.decorator_list:
            if isinstance(decorator, ast.Call):
                name = decorator.func.attr if isinstance(decorator.func, ast.Attribute) else decorator.func.id
            else:
                name = decorator.attr if isinstance(decorator, ast.Attribute) else decorator.id
            decorator_names.append(name)

        if decorator_name in decorator_names:
            method_names.append(method.name)
    return [getattr(class_, n) for n in method_names]


def plot_data_display(selector: str = None, load_by_default: bool = False, serialize_data: bool = True):
    """ Decorator to plot data."""
    def decorator(function):
        """ Decorator to plot data."""
        set_decorated_function_metadata(function=function, type_="plot_data", selector=selector,
                                        serialize_data=serialize_data, load_by_default=load_by_default)
        return function
    return decorator


def markdown_display(selector: str = None, load_by_default: bool = False, serialize_data: bool = False):
    """ Decorator to markdown."""
    def decorator(function):
        """ Decorator to markdown. """
        set_decorated_function_metadata(function=function, type_="markdown", selector=selector,
                                        serialize_data=serialize_data, load_by_default=load_by_default)
        return function
    return decorator


def cad_display(selector: str = None, load_by_default: bool = False, serialize_data: bool = True):
    """ Decorator to markdown."""
    def decorator(function):
        """ Decorator to markdown. """
        set_decorated_function_metadata(function=function, type_="cad", selector=selector,
                                        serialize_data=serialize_data, load_by_default=load_by_default)
        return function
    return decorator


def set_decorated_function_metadata(function, type_: str, selector: str = None,
                                    serialize_data: bool = False, load_by_default: bool = False):
    """ Attach metadata to function object. Is there any better way to do this ? It seems a bit dirty. """
    setattr(function, "type_", type_)
    setattr(function, "selector", selector)
    setattr(function, "serialize_data", serialize_data)
    setattr(function, "load_by_default", load_by_default)
