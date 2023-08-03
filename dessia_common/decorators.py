""" Provides decorators that work as 'flags' for display settings. """

from typing import Type, List, TypeVar, Generic, Dict, Any, Tuple
from dessia_common.utils.helpers import full_classname, get_python_class_from_class_name
import inspect
import ast
import textwrap


DISPLAY_DECORATORS = ["plot_data_view", "markdown_view", "cad_view"]


def get_all_decorated_methods(class_: Type) -> List[ast.FunctionDef]:
    """ Get all decorated method from class_. """
    methods = inspect.getmembers(class_, inspect.isfunction) + inspect.getmembers(class_, inspect.ismethod)
    function_defs = []
    for _, method in methods:
        source = textwrap.dedent(inspect.getsource(method))
        method_tree = ast.parse(source)
        function_defs.append(method_tree.body[0])
    return [m for m in function_defs if isinstance(m, ast.FunctionDef) and m.decorator_list]


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


def plot_data_view(selector: str = None, load_by_default: bool = False, serialize_data: bool = True):
    """ Decorator to plot data."""
    def decorator(function):
        """ Decorator to plot data."""
        set_decorated_function_metadata(function=function, type_="plot_data", selector=selector,
                                        serialize_data=serialize_data, load_by_default=load_by_default)
        return function
    return decorator


def markdown_view(selector: str = None, load_by_default: bool = False, serialize_data: bool = False):
    """ Decorator to markdown."""
    def decorator(function):
        """ Decorator to markdown. """
        set_decorated_function_metadata(function=function, type_="markdown", selector=selector,
                                        serialize_data=serialize_data, load_by_default=load_by_default)
        return function
    return decorator


def cad_view(selector: str = None, load_by_default: bool = False, serialize_data: bool = True):
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


T = TypeVar("T")


class CadViewSelector(Generic[T]):
    """ Typing that denotes a method of class T. """

    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name

    def __deepcopy__(self, memo=None):
        return CadViewSelector(self.class_, self.name)

    def get_method(self):
        """ Helper to get real method from class_ and method name. """
        return getattr(self.class_, self.name)

    def to_dict(self):
        """ Write Method Type as a dictionary. """
        classname = full_classname(object_=self.class_, compute_for='class')
        method_type_classname = full_classname(object_=self.__class__, compute_for="class")
        return {"class_": classname, "name": self.name, "object_class": method_type_classname}

    @classmethod
    def dict_to_object(cls, dict_) -> 'CadViewSelector':
        """ Deserialize dictionary as a Method Type. """
        class_ = get_python_class_from_class_name(dict_["class_"])
        return cls(class_=class_, name=dict_["name"])
