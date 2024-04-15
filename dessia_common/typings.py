""" Typing for dessia_common. """
import inspect
from typing import TypeVar, Generic, Dict, Any, Tuple, get_type_hints

from dessia_common.utils.helpers import full_classname, get_python_class_from_class_name


T = TypeVar('T')


class Subclass(Generic[T]):
    """ Typing that denotes a Subclass of T. """


class InstanceOf(Generic[T]):
    """ Typing that denotes a Instance of T. """


class MethodType(Generic[T]):
    """ Typing that denotes a method of class T. """

    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name

    def __deepcopy__(self, memo=None):
        return MethodType(self.class_, self.name)

    def get_method(self):
        """ Helper to get real method from class_ and method name. """
        return getattr(self.class_, self.name)

    def output_type(self):
        """ Helper to get method output type. """
        method = self.get_method()
        hints = get_type_hints(method)
        return hints.get("return", None)

    def to_dict(self):
        """ Write Method Type as a dictionary. """
        classname = full_classname(object_=self.class_, compute_for='class')
        method_type_classname = full_classname(object_=self.__class__, compute_for="class")
        return {"class_": classname, "name": self.name, "object_class": method_type_classname}

    @classmethod
    def dict_to_object(cls, dict_) -> 'MethodType':
        """ Deserialize dictionary as a Method Type. """
        class_ = get_python_class_from_class_name(dict_["class_"])
        return cls(class_=class_, name=dict_["name"])


class ClassMethodType(MethodType[T]):
    """ Typing that denotes a class method of class T. """
    
    def __init__(self, class_: T, name: str):
        MethodType.__init__(self, class_=class_, name=name)


class AttributeType(Generic[T]):
    """ Typing that denotes an attribute of class T. """

    def __init__(self, class_: T, name: str):
        self.class_ = class_
        self.name = name

    def __deepcopy__(self, memo=None):
        return AttributeType(self.class_, self.name)

    @property
    def type_(self):
        """ Get the user-defined type of the attribute."""
        parameters = inspect.signature(self.class_).parameters
        attribute = parameters.get(self.name, None)
        if not attribute:
            return None
        if not hasattr(attribute, "annotation"):
            return attribute
        if attribute.annotation == inspect.Parameter.empty:
            return None
        return attribute.annotation

    def to_dict(self):
        """ Write Attribute Type as a dictionary. """
        classname = full_classname(object_=self.class_, compute_for='class')
        method_type_classname = full_classname(object_=self.__class__, compute_for="class")
        return {"class_": classname, "name": self.name, "object_class": method_type_classname}

    @classmethod
    def dict_to_object(cls, dict_) -> 'AttributeType':
        """ Deserialize dictionary as an Attribute Type. """
        class_ = get_python_class_from_class_name(dict_["class_"])
        return cls(class_=class_, name=dict_["name"])


class ClassAttributeType(AttributeType[T]):
    """ Typing that denotes a class attribute of class T. """

    def __init__(self, class_: T, name: str):
        AttributeType.__init__(self, class_=class_, name=name)


class ViewType(AttributeType[T]):
    """ Typing that denotes a Display Settings. """

    decorator = None

    def get_method(self):
        """ Helper to get real method from class_ and method name. """
        settings = self.class_._display_settings_from_selector(self.name)
        return getattr(self.class_, settings.method)


class CadViewType(ViewType[T]):
    """ Typing that denotes a CAD Display Settings. """

    decorator = "cad_view"


class MarkdownType(ViewType[T]):
    """ Typing that denotes a CAD Display Settings. """

    decorator = "markdown_view"


class PlotDataType(ViewType[T]):
    """ Typing that denotes a CAD Display Settings. """

    decorator = "plot_data_view"


# Types Aliases
JsonSerializable = Dict[str, Any]
RGBColor = Tuple[float, float, float]
