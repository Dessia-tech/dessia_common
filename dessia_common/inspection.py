"""Module to inspect functions and classes."""

import inspect
import re
import hashlib

from typing import Dict


_REGEX = {"def_function": r"(?<=def )\w+", #def functions r"(\.\w|\w)+(?=\(.+?\)\:)"
          "all_functions": r"[\w+\.\[\](\)\"']+(?=\()",
          "not_def_1": r"(?<!def )",
          "not_def_2": r"(?=\()",
          "return": r"(?<=return ).*(?=(\\n)?)"}

class Function:
    """Class to store a function and its inputs, outputs and modifications."""

    def __init__(self, function):
        self.function = function
        self.files = inspect.getfile(self.function)
        self.hashes = [self.hash_function]
        self.sourcelines = inspect.getsourcelines(self.function)
        self.inputs = self.find_inputs()
        self.outputs = self.find_outputs()
        self.modifications = self.find_modifications()
        self.used_functions = self.find_used_functions()

    def find_used_functions(self) -> Dict:
        """Find all functions used in the function."""
        used_functions = {"called" : {},
                          "defined" : {},
                          "object" : {}} #object is just initialized, it'll be filled later
        for line in self.sourcelines[0]:
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
                hash_= function_to_add.hash_function
                self.hashes.append(hash_)
            if re.search(_REGEX["def_function"],line):
                if hash_ in [self.hashes]:
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
        outputs = {}
        if inspect.signature(self.function).return_annotation == inspect._empty:
            outputs["type"] = ""
        else:
            outputs["type"] = inspect.signature(self.function).return_annotation
        for line in self.sourcelines[0]:
            reg = re.search(_REGEX["return"], line)
            if reg:
                try:
                    outputs["return"].append(reg.group())
                except KeyError:
                    outputs["return"] = [reg.group()]
        return outputs

    def find_modifications(self):
        """Find all modifications of the function."""
        modifications = {}
        return modifications

    @property
    def hash_function(self) -> int:
        """Hash function."""
        try:
            my_bytes = f"{inspect.getfile(self.function)}{self.function.__name__}".encode('utf-8')
        except NameError:
            print("NameError")
            return 0
        hash_object = hashlib.blake2b(digest_size=3)
        hash_object.update(my_bytes)
        return int.from_bytes(hash_object.digest(), byteorder='big')


def get_function_by_name(function_name):
    """Get function by name."""
    current_frame = inspect.currentframe()
    name = function_name.rsplit('.')[-1]
    print(name)
    i=0
    while current_frame:
        i+=1
        for _, obj in current_frame.f_locals.items():
            if inspect.isfunction(obj) and obj.__name__ == name:
                if inspect.isbuiltin(obj):
                    return function_name
                return obj
        for _, obj in current_frame.f_globals.items():
            if inspect.isfunction(obj) and obj.__name__ == name:
                if inspect.isbuiltin(obj):
                    return function_name
                return obj
        current_frame = current_frame.f_back
    return function_name


def escape_regex(string):
    """Escape regex special characters in a string."""
    special_chars = r'[\\.*+?|(){}\[\]^$]'
    return re.sub(special_chars, r'\\\g<0>', string)
