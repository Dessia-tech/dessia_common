from typing import Union, Type
from dessia_common.schemas.core import deserialize_annotation, extract_args
from dessia_common.forms import StandaloneObject
from dessia_common.measures import Distance
from dessia_common.typings import Subclass, InstanceOf, MethodType, ClassMethodType
from dessia_common.files import StringFile, BinaryFile

# Test args extraction
assert extract_args("int") == ["int"]
assert extract_args("dessia_common.forms.StandaloneObject") == ["dessia_common.forms.StandaloneObject"]
assert extract_args("list[dessia_common.forms.StandaloneObject]") == ["list[dessia_common.forms.StandaloneObject]"]
assert extract_args("list[list[dessia_common.forms.StandaloneObject]]") == ["list[list[dessia_common.forms.StandaloneObject]]"]
assert extract_args("int, dessia_common.forms.StandaloneObject") == ["int", "dessia_common.forms.StandaloneObject"]
assert extract_args("int, str") == ["int", "str"]
assert extract_args("int,tuple[dessia_common.forms.StandaloneObject,str]") == ["int", "tuple[dessia_common.forms.StandaloneObject,str]"]
assert extract_args("tuple[dessia_common.forms.StandaloneObject,str], int") == ["tuple[dessia_common.forms.StandaloneObject,str]", "int"]
assert extract_args("int,tuple[dessia_common.forms.StandaloneObject, str]") == ["int", "tuple[dessia_common.forms.StandaloneObject,str]"]
assert extract_args("int, tuple[int, tuple[tuple[int, float], int]], int") == ["int", "tuple[int,tuple[tuple[int,float],int]]", "int"]
assert extract_args("int, tuple[int, tuple[tuple[int, float], int]], int,list[package.module.class]") == ["int", "tuple[int,tuple[tuple[int,float],int]]", "int", "list[package.module.class]"]
assert extract_args("int, tuple[int, tuple[tuple[int, float], int]], int,list[package.module.class], int") == ["int", "tuple[int,tuple[tuple[int,float],int]]", "int", "list[package.module.class]", "int"]

# Simple types
assert deserialize_annotation("int") == int
assert deserialize_annotation("str") == str
assert deserialize_annotation("float") == float
assert deserialize_annotation("dessia_common.forms.StandaloneObject") == StandaloneObject
assert deserialize_annotation("dessia_common.measures.Distance") == Distance

# Tuple
assert deserialize_annotation("tuple[int]") == tuple[int]
assert deserialize_annotation("tuple[int]") == tuple[int]
assert deserialize_annotation("tuple[int, str]") == tuple[int, str]
assert deserialize_annotation("tuple[int, ...]") == tuple[int, ...]
assert deserialize_annotation("tuple[int, dessia_common.forms.StandaloneObject]") == tuple[int, StandaloneObject]
assert deserialize_annotation("tuple[int, tuple[dessia_common.forms.StandaloneObject, str]]") == tuple[int, tuple[StandaloneObject, str]]

# Union
assert deserialize_annotation("Union[dessia_common.forms.StandaloneObject, int]") == Union[StandaloneObject, int]

# List
assert deserialize_annotation("list[int]") == list[int]
assert deserialize_annotation("list[dessia_common.forms.StandaloneObject]") == list[StandaloneObject]
assert deserialize_annotation("list[list[int]]") == list[list[int]]
assert deserialize_annotation("list[tuple[dessia_common.forms.StandaloneObject, int]]") == list[tuple[StandaloneObject, int]]
assert deserialize_annotation("list[tuple[int, tuple[dessia_common.forms.StandaloneObject, int], dessia_common.measures.Distance]]") == list[tuple[int, tuple[StandaloneObject, int], Distance]]

# Dict
assert deserialize_annotation("dict[str, int]") == dict[str, int]
assert deserialize_annotation("dict[str, tuple[int, float]]") == dict[str, tuple[int, float]]

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
