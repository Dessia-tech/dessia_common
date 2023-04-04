import ast
import inspect
from dessia_common.tests import Car
from dessia_common.core import DessiaObject
from typing import TypeVar, Generic
import re

REGEX ={
    'class_methods': [r'(@classmethod\\n', r"( {4}def )\w+(?=\()"] ,
    'methods': [r'\\n', r"( {4}def )\w+(?=\()"] ,
    'attribute_class': [r''],
    'attribute_instance': [r"( {4}self\.)\w+(?=\s*=)"],
    'test': [r'@classmethod\s+def\s+(\w+)\s*\('],
    '2': [r''],
    '3': [r''],
}

class UnrealClassForTesting(DessiaObject):
    def __init__(self, int_: int , variable_, float_: float = 0.0, str_: str = 'test', bool_: bool = True,
                  list_: list = [1,2,3], dict_: dict = {'a':1,'b':2}, two: int = 2, name: str ='', **kwargs):
        self.int_ = int_
        variable_= variable_
        self.float_ = float_
        self.str_ = str_
        self.bool_ = bool_
        self.list_ = list_
        self.dict_ = dict_
        self.two = two
        self.name = name
        self.combine : int = self.int_ + self.two
        DessiaObject.__init__(self, **kwargs)

    def add_attribute_to_self(self, tuple_: tuple = (0,0), list_of_tuple: list = [(0,0),(1,1)]):
        self.tuple_ = tuple_
        self.list_of_tuple = list_of_tuple


class ModuleToText():
    """Module to get the text of a class.
    
    :param name: Name of the Module.
    """

    def __init__(self, name : str):
        self.name = name
        self.text_informations = self.get_module_text()
    
    def get_module_text(self) -> str:
        return inspect.getsource(self.name)
    


class ClassToText():
    """Class to get the text of a class.
    
    :param name: Name of the class.
    :param upper_class: If True, get the text of the class and its superclasses.
    """

    def __init__(self, name : str, upper_class : bool = False):
        self.name = name
        if upper_class:
            self.text_informations = ClassToText.get_class_and_super_class_text(self.name)
        else:
            self.text_informations = ClassToText.get_class_text(self.name)
    
    @classmethod
    def get_class_text(cls, class_name) -> str:
        return inspect.getsourcelines(class_name)
    
    @classmethod
    def get_class_and_super_class_text(cls, class_name) -> str:
        text_informations =([],0)
        for c in inspect.getmro(class_name):
            if c.__name__!='object':
                source_lines = inspect.getsourcelines(c)
                text_informations[0].extend(source_lines[0])
                text_informations[1] += source_lines[1]
        return text_informations


class RegexContainer():
    """Class to store a regex and apply it to a string.
    
    :param regex: Regex to apply.
    :param nom: Name of the regex.
    """

    def __init__(self, regex : str, nom : str = ''):
        self.regex = regex
        self.nom = nom
    
    def apply_regex(self, chaine, nb : int = 1):
        matches = re.findall(self.regex, chaine)
        return matches


class GetAttributesFromCode():
    """Class to get the attributes of a class from its text.
    note: the attributes of the superclasses are not included.
    
    :param class_name: Name of the class.
    :param all_attributs: If True, get the attributes of the class and its superclasses.
    :param name: Name of the class.
    """

    def __init__(self, class_name : str, all_attributs : bool = False, name : str = '') -> None:
        self.text_informations = ClassToText(class_name, upper_class=all_attributs).text
        self.attributes = RegexContainer(REGEX['attribute_instance'], 'Attribute').apply_regex(self.text)
        self.name = name


#class GetClassAttributesFromCode():


class GetClassMethodsFromCode():
    '''Class to get the class methods of a class from its text.
    note: the class methods of the superclasses are not included.
    
    :param class_name: Name of the class.
    :param all_class_methods: If True, get the class methods of the class and its superclasses.
    :param name: Name of the class.
    '''

    def __init__(self, class_name : str, all_class_methods : bool = False, name : str = '') -> None:
        self.text = ClassToText(class_name, upper_class=all_class_methods).text
        self.class_methods = RegexContainer(REGEX['class_methods'], 'ClassMethod').apply_regex(self.text)


class GetMethodsFromCode():
    """Class to get the methods of a class from its text.
    note: the methods of the superclasses are not included.

    :param class_name: Name of the class.
    :param all_methods: If True, get the methods of the class and its superclasses.
    :param name: Name of the class.
    """

    def __init__(self, class_name: str, all_methods : bool = False, name : str = ''):
        self.text = ClassToText(class_name, upper_class=all_methods).text
        all = RegexContainer(REGEX['methods'], 'Method').apply_regex(self.text)
        selected = []
        classmethods = RegexContainer(REGEX['class_methods'], 'ClassMethod').apply_regex(self.text)
        for method in all:
            if method not in classmethods:
                selected.append(method)
        self.methods = selected
        self.name = name


class GetTestFromCode():
    """Class to get the methods of a class from its text.
    note: the methods of the superclasses are not included.

    :param class_name: Name of the class.
    :param all_methods: If True, get the methods of the class and its superclasses.
    :param name: Name of the class.
    """

    def __init__(self, class_name: str, all_methods : bool = False, name : str = ''):
        self.text = ClassToText(class_name, upper_class=all_methods).text
        all = RegexContainer(REGEX['test'], 'Method').apply_regex(self.text)
        self.results = all
        self.name = name




"""class AttributesFinder(ast.NodeVisitor):
    def __init__(self):
        self.self_attrs = set()
        
    def visit_Assign(self, node):
        print (node.targets[0], '   ',isinstance(node.targets[0], ast.Attribute) and isinstance(node.targets[0].value, ast.Name))
        if isinstance(node.targets[0], ast.Attribute) and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == 'self':
            self_attr = node.targets[0].attr
            self.self_attrs.add(self_attr)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        if node.name == '__copy__' or node.name == '__deepcopy__':
            for arg in node.args.args:
                if arg.arg != 'self':
                    self.self_attrs.add(arg.arg)
        self.generic_visit(node)

class GetAttributesFromCode():
    def __init__(self, class_name: str):
        self.parsed_ast = ast.parse(ClassToText(class_name).text)
        self.attributes = GetAttributesFromCode.get_attributes(self.parsed_ast)

    @classmethod
    def get_attributes(cls, parsed_ast) -> list:
        finder = AttributesFinder()
        finder.visit(parsed_ast)
        return finder.self_attrs

#print(find_self_attrs(a))

class MyVisitor(ast.NodeVisitor):
    def visit_Str(self, node):
        # traitement pour les nœuds de type Str
        pass

    def visit_Num(self, node):
        # traitement pour les nœuds de type Num
        pass

    def visit_BinOp(self, node):
        # traitement pour les nœuds de type BinOp
        pass

    # Ajouter d'autres méthodes pour traiter d'autres types de nœuds
code = "print('Hello, world!')"
tree = ast.parse(code)

my_visitor = MyVisitor()
my_visitor.visit(tree)"""