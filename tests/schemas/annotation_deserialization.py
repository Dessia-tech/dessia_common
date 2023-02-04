from typing import Union, Type, List, Dict, Tuple
from dessia_common.schemas.core import deserialize_annotation, extract_args
from dessia_common.forms import StandaloneObject
from dessia_common.measures import Distance
from dessia_common.typings import Subclass, InstanceOf, MethodType, ClassMethodType
from dessia_common.files import StringFile, BinaryFile

# Test args extraction
assert extract_args("int") == ["int"]
assert extract_args("dessia_common.forms.StandaloneObject") == ["dessia_common.forms.StandaloneObject"]
assert extract_args("List[dessia_common.forms.StandaloneObject]") == ["List[dessia_common.forms.StandaloneObject]"]
assert extract_args("List[List[dessia_common.forms.StandaloneObject]]") == ["List[List[dessia_common.forms.StandaloneObject]]"]
assert extract_args("int, dessia_common.forms.StandaloneObject") == ["int", "dessia_common.forms.StandaloneObject"]
assert extract_args("int, str") == ["int", "str"]
assert extract_args("int,Tuple[dessia_common.forms.StandaloneObject,str]") == ["int", "Tuple[dessia_common.forms.StandaloneObject,str]"]
assert extract_args("Tuple[dessia_common.forms.StandaloneObject,str], int") == ["Tuple[dessia_common.forms.StandaloneObject,str]", "int"]
assert extract_args("int,Tuple[dessia_common.forms.StandaloneObject, str]") == ["int", "Tuple[dessia_common.forms.StandaloneObject,str]"]
assert extract_args("int, Tuple[int, Tuple[Tuple[int, float], int]], int") == ["int", "Tuple[int,Tuple[Tuple[int,float],int]]", "int"]
assert extract_args("int, Tuple[int, Tuple[Tuple[int, float], int]], int,List[package.module.class]") == ["int", "Tuple[int,Tuple[Tuple[int,float],int]]", "int", "List[package.module.class]"]
assert extract_args("int, Tuple[int, Tuple[Tuple[int, float], int]], int,List[package.module.class], int") == ["int", "Tuple[int,Tuple[Tuple[int,float],int]]", "int", "List[package.module.class]", "int"]

# Simple types
assert deserialize_annotation("int") == int
assert deserialize_annotation("str") == str
assert deserialize_annotation("float") == float
assert deserialize_annotation("dessia_common.forms.StandaloneObject") == StandaloneObject
assert deserialize_annotation("dessia_common.measures.Distance") == Distance

# Tuple
assert deserialize_annotation("Tuple[int]") == Tuple[int]
assert deserialize_annotation("Tuple[int]") == Tuple[int]
assert deserialize_annotation("Tuple[int, str]") == Tuple[int, str]
assert deserialize_annotation("Tuple[int, ...]") == Tuple[int, ...]
assert deserialize_annotation("Tuple[int, dessia_common.forms.StandaloneObject]") == Tuple[int, StandaloneObject]
assert deserialize_annotation("Tuple[int, Tuple[dessia_common.forms.StandaloneObject, str]]") == Tuple[int, Tuple[StandaloneObject, str]]

# Union
assert deserialize_annotation("Union[dessia_common.forms.StandaloneObject, int]") == Union[StandaloneObject, int]

# List
assert deserialize_annotation("List[int]") == List[int]
assert deserialize_annotation("List[dessia_common.forms.StandaloneObject]") == List[StandaloneObject]
assert deserialize_annotation("List[List[int]]") == List[List[int]]
assert deserialize_annotation("List[Tuple[dessia_common.forms.StandaloneObject, int]]") == List[Tuple[StandaloneObject, int]]
assert deserialize_annotation("List[Tuple[int, Tuple[dessia_common.forms.StandaloneObject, int], dessia_common.measures.Distance]]") == List[Tuple[int, Tuple[StandaloneObject, int], Distance]]

# Dict
assert deserialize_annotation("Dict[str, int]") == Dict[str, int]
assert deserialize_annotation("Dict[str, Tuple[int, float]]") == Dict[str, Tuple[int, float]]

# InstanceOf
assert deserialize_annotation("InstanceOf[dessia_common.forms.StandaloneObject]") == InstanceOf[StandaloneObject]

# Subclass
assert deserialize_annotation("Subclass[dessia_common.forms.StandaloneObject]") == Subclass[StandaloneObject]

# Methods
assert deserialize_annotation("MethodType[dessia_common.forms.StandaloneObject]") == MethodType[StandaloneObject]
assert deserialize_annotation("ClassMethodType[dessia_common.forms.StandaloneObject]") == ClassMethodType[StandaloneObject]

# Types
assert deserialize_annotation("Type") == Type
assert deserialize_annotation("Type[dessia_common.forms.StandaloneObject]") == Type[StandaloneObject]

# Files
assert deserialize_annotation("dessia_common.files.StringFile") == StringFile
assert deserialize_annotation("dessia_common.files.BinaryFile") == BinaryFile

print("script 'annotation_deserialization' has passed.")
