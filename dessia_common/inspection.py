"""Module to inspect functions and classes"""

import inspect
import re
from typing import List, Dict
from deepdiff import DeepDiff
import hashlib


_REGEX = {"def_function": r"(?<=def )\w+", #def funcionts r"(\.\w|\w)+(?=\(.+?\)\:)"
          "all_functions": r"(\w*\.?\[(\"|')\w+(\"|')\]\.?|\w)+\w+\.?\w*(?=\()",
          "not_def_1": r"(?<!def )",
          "not_def_2": r"(?=\()",
          "return": r"(?<=return )(\w*\.?\[(\"|')\w+(\"|')\]\.?|\w)*\w+\.?\w*(\(.*\))*"}

def test(a: int, b: List[int], c: Dict[str, int]) -> int:
    """Test function"""

    return a + b[0] + c['a']

def escape_regex(string):
    """Escape regex special characters in a string"""
    special_chars = r'[\\.*+?|(){}\[\]^$]'
    return re.sub(special_chars, r'\\\g<0>', string)

class Function:
    """Class to store a function and its inputs, outputs and modifications"""

    functions = []

    def __init__(self, function, is_called=False, is_def=False):
        self.function = function
        self.name = function.__name__
        self.files = inspect.getfile(self.function)
        self.hash_ = hash_function(self.function)
        self.hashes = [self.hash_]
        self.sourcelines = inspect.getsourcelines(self.function)
        self.signature = inspect.signature(self.function)
        self.is_called = is_called
        self.is_def = is_def
        if is_called or is_def:
            self.is_object = False
        else:
            self.is_object = True
        self.inputs = self.find_inputs()
        self.outputs = self.find_outputs()
        self.modifications = self.find_modifications()
        self.used_functions = self.find_used_functions()

    def find_used_functions(self) -> Dict:
        """Find all functions used in the function"""
        used_functions = {"called" : {},
                          "defined" : {},
                          "object" : {}} #object is just initialised, it'll be filled later
        for line in self.sourcelines[0]:
            reg=re.search(_REGEX["all_functions"], str(line))
            if not reg:
                continue
            if re.search(_REGEX["def_function"],line):
                if reg.group()==self.name:
                    continue
                function_to_add = Function(globals()[reg.group()], is_def=True)
                if not self.is_there(function_to_add):
                    print("Already added")
                    continue
                try:
                    used_functions["defined"][reg.group()].append(line)
                except KeyError:
                    used_functions["defined"][reg.group()] = [function_to_add]
            elif re.search(_REGEX["not_def_1"]+escape_regex(reg.group())+_REGEX["not_def_2"],line):
                try:
                    used_functions["called"][reg.group()].append(line)
                except KeyError:
                    used_functions["called"][reg.group()] = [line]
            else:
                try:
                    used_functions["object"][reg.group()].append(line)
                except KeyError:
                    used_functions["object"][reg.group()] = [line]
        return used_functions
        
    def find_inputs(self):
        """Find all inputs of the function"""
        return dict(self.signature.parameters)

    def find_outputs(self):
        """Find all outputs of the function"""
        outputs = {}
        if self.signature.return_annotation == inspect._empty:
            outputs["type"] = ""
        else:
            outputs["type"] = self.signature.return_annotation
        for line in self.sourcelines[0]:
            reg = re.search(_REGEX["return"], line)
            if reg:
                try:
                    outputs["return"].append(reg.group())
                except KeyError:
                    outputs["return"] = [reg.group()]
        return outputs

    def find_modifications(self):
        """Find all modifications of the function"""
        modifications = {}
        return modifications

    def test_function(self):
        """Compare two functions"""
        diff_getmembers(self.function, self.function)
        a = 1
        b = 2
        c = a + b
        return c
    
    def is_there(self, function) -> bool:
        """Add a function to the list of functions"""
        is_there = False
        function_hash = hash_function(function)
        if function_hash in self.hashes:
            is_there = True
        else:
            self.hashes.append(function_hash)
        return is_there


def diff_getmembers(oldfunction, newfunction):
    for el1 in inspect.getmembers(oldfunction):
        for el2 in inspect.getmembers(newfunction):
            if el1[0] == el2[0]:
                print(f"Parametre : {el1[0]}")
                print(f"{DeepDiff(el1[1], el2[1])} \n")

def inspect_object(object_):
    """Inspect an object"""
    print("uifbfbzzbjenzeb")
    if inspect.ismodule(object_):
        print("ismodule")
    if inspect.isclass(object_):
        print("isclass")
    if inspect.isfunction(object_):
        print("isfunction")
    if inspect.ismethod(object_):
        print("ismethod")
    if inspect.ismethoddescriptor(object_):
        print("ismethoddescriptor")
    if inspect.isdatadescriptor(object_):
        print("isdatadescriptor")
    if inspect.isgeneratorfunction(object_):
        print("isgeneratorfunction")
    if inspect.isgenerator(object_):
        print("isgenerator")
    if inspect.iscoroutinefunction(object_):
        print("iscoroutinefunction")
    if inspect.iscoroutine(object_):
        print("iscoroutine")
    if inspect.isasyncgenfunction(object_):
        print("isasyncgenfunction")
    if inspect.isasyncgen(object_):
        print("isasyncgen")
    if inspect.istraceback(object_):
        print("istraceback")
    if inspect.isframe(object_):
        print("isframe")
    if inspect.iscode(object_):
        print("iscode")
    if inspect.isbuiltin(object_):
        print("isbuiltin")
    if inspect.isroutine(object_):
        print("isroutine")

def hash_function(function) -> int:
    """Hash function"""
    try:
        my_bytes = f"{inspect.getfile(function)}{function.__name__}".encode('utf-8')
    except NameError:
        print("NameError")
        return 0
    hash_object = hashlib.blake2b(digest_size=3)
    hash_object.update(my_bytes)
    return int.from_bytes(hash_object.digest(), byteorder='big')