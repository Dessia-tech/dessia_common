"""Module to inspect functions and classes."""

import inspect
import re
import hashlib

from typing import Dict


_REGEX = {"def_function": r"(?<=def )\w+", #def functions r"(\.\w|\w)+(?=\(.+?\)\:)"
          "all_functions": r"[\w+\.\[\](\)\"']+(?=\()",
          "not_def_1": r"(?<!def )",
          "not_def_2": r"(?=\()",
          "return": r"(?<=return ).*(?=(\\n)?)",
          "descriptors": r"(?<=\@)\w+",
          "wrong_functions": r"[\w+\.\[\](\)\"']+(?=\()(?!\(self|\(cls)"
          }

class Function():
    """Class to store a function and its inputs, outputs and modifications."""

    def __init__(self, function):
        self.function = function
        self.hashes = []
        self.inputs = self.find_inputs()
        self.outputs = self.find_outputs()
        self.modifications = self.find_modifications()
        self.used_functions = self.find_used_functions()

    def find_used_functions(self) -> Dict:
        """Find all functions used in the function."""
        used_functions = {
            "called" : {},
            "defined" : {},
            "object" : {},
            "lambda" : {} #object is just initialized, it'll be filled later
        }
        for line in inspect.getsourcelines(self.function)[0]:
            reg=re.search(_REGEX["all_functions"], str(line))
            if not reg:
                continue
            if reg.group()==self.function.__name__:
                continue
            to_try = get_function_by_name(reg.group())
            if isinstance(to_try, str):
                function_to_add = line
            else:
                function_to_add = Function(to_try)
                self.hashes.append(function_to_add.hash_)
            if re.search(_REGEX["def_function"],line):
                if function_to_add.hash_ in [self.hashes]:
                    print("Already added")
                    continue
                try:
                    used_functions["defined"][reg.group()].append(function_to_add)
                except KeyError:
                    used_functions["defined"][reg.group()] = [function_to_add]
            elif re.search(_REGEX["not_def_1"]+escape_regex(reg.group())+_REGEX["not_def_2"],line):
                try:
                    used_functions["called"][reg.group()].append(function_to_add)
                except KeyError:
                    used_functions["called"][reg.group()] = [function_to_add]
            else:
                try:
                    used_functions["object"][reg.group()].append(function_to_add)
                except KeyError:
                    used_functions["object"][reg.group()] = [function_to_add]
        return used_functions

    def find_inputs(self):
        """Find all inputs of the function."""
        return dict(inspect.signature(self.function).parameters)

    def find_outputs(self):
        """Find all outputs of the function."""
        outputs = {"type" : {},
                   "return" : {}}
        if inspect.signature(self.function).return_annotation == inspect._empty:
            outputs["type"] = ""
        else:
            outputs["type"] = inspect.signature(self.function).return_annotation
        for line in inspect.getsourcelines(self.function)[0]:
            reg = re.search(_REGEX["return"], line)
            if reg:
                outputs["return"][reg.group()] = line
        return outputs

    def find_modifications(self):
        """Find all modifications of the function."""
        modifications = {}
        return modifications

    @property
    def hash_(self) -> int:
        """Hash function."""
        try:
            my_bytes = f"{inspect.getfile(self.function)}{self.function.__name__}".encode('utf-8')
        except NameError:
            print("NameError")
            return 0
        hash_object = hashlib.blake2b(digest_size=3)
        hash_object.update(my_bytes)
        return int.from_bytes(hash_object.digest(), byteorder='big')


class Method(Function):
    """Class to store a method and its inputs, outputs and modifications."""
    def __init__(self, function, class_):
        super().__init__(function)
        self.class_ = class_


class Class():
    """"Class to store a class and its methods and attributes."""
    def __init__(self, class_):
        self.class_ = class_
        self.super_classes = self.class_.__bases__
        self.hashes = []
        self.methods = self.find_methods() #return a dictionary of methods
        self.attributes = self.find_attributes() #return a dictionary of attributes
        

    def find_methods(self):
        """Find all methods of the class."""
        methods = {
            "__init__" : None,
            "methods" : {},
            "decorated" : {},
            "wrong_functions" : {}
        }
        decorators_names = []
        wrong_functions_names = []        
        for line in inspect.getsourcelines(self.class_)[0]:
            descriptors_reg=re.search(_REGEX["descriptors"], str(line))
            wrong_functions_reg=re.search(_REGEX["wrong_functions"], str(line))
            if descriptors_reg:
                decorators_names.append(descriptors_reg.group())
            if wrong_functions_reg:
                wrong_functions_names.append(wrong_functions_reg.group())
        for name in self.class_.__dict__.keys():
            type_ = re.search(r"(?<=').+(?=')",str(type(self.class_.__dict__[name]))).group()
            if name == '__init__':
                methods["__init__"] = Method(self.class_.__dict__[name], self.class_)
            elif type_ in ("function", "method"):
                methods["methods"][name] = Method(self.class_.__dict__[name], self.class_)
            elif type_ in decorators_names:
                methods["decorated"][type_][name] = Method(self.class_.__dict__[name], self.class_)
            elif name in wrong_functions_names:
                methods["wrong_functions"][name] = Method(self.class_.__dict__[name], self.class_)
        return methods

    def find_attributes(self):
        """Find all attributes of the class."""
        attributes = {
            "instance attributes" : {},
            "class attributes" : {},
            "wrong attributes" : {}
        }

        return attributes

    @property
    def hash_(self) -> int:
        """Hash class."""
        try:
            my_bytes = f"{inspect.getfile(self.class_)}{self.class_.__name__}".encode('utf-8')
        except NameError:
            print("NameError")
            return 0
        hash_object = hashlib.blake2b(digest_size=3)
        hash_object.update(my_bytes)
        return int.from_bytes(hash_object.digest(), byteorder='big')


def get_function_by_name(function_name : str):
    """Get function by name."""
    current_frame = inspect.currentframe()
    name = function_name.rsplit('.')[-1]
    result = function_name
    print(name)
    while current_frame or not isinstance(result, str):
        for _, obj in current_frame.f_locals.items():
            if inspect.isfunction(obj) and obj.__name__ == name:
                if not inspect.isbuiltin(obj):
                    result = obj
                    break
        for _, obj in current_frame.f_globals.items():
            if inspect.isfunction(obj) and obj.__name__ == name:
                if not inspect.isbuiltin(obj):
                    result = obj
                    break
        current_frame = current_frame.f_back
    return result

def escape_regex(string):
    """Escape regex special characters in a string."""
    special_chars = r'[\\.*+?|(){}\[\]^$]'
    return re.sub(special_chars, r'\\\g<0>', string)

print(type(get_function_by_name('Generator.generate')))
