#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utilities for workflows. """

from typing import List, Dict

from dessia_common.schemas.core import get_schema, SchemaAttribute
from dessia_common.serialization import SerializableObject
from dessia_common.utils.types import is_file_or_file_sequence


class ToScriptElement:
    """ Class meant to improve to_script readability. """

    def __init__(self, declaration: str, before_declaration: str = None,
                 imports: List[str] = None, imports_as_is: List[str] = None):
        self.before_declaration = before_declaration
        self.declaration = declaration
        self.imports = imports
        self.imports_as_is = imports_as_is

    def imports_to_str(self) -> str:
        """  Import a script as str. """
        script_imports = ""
        for module, class_list in self.get_import_dict().items():
            script_imports += f"from {module} import {', '.join(class_list)}\n"

        for import_as_is in self.imports_as_is:
            import_str = f"import {import_as_is}\n"
            if import_str not in script_imports:
                script_imports += import_str

        return script_imports

    def get_import_dict(self) -> Dict[str, List[str]]:
        """ Get imported modules of script into a dict. """
        imports_dict: Dict[str, List[str]] = {}
        for import_ in self.imports:
            module = '.'.join(import_.split('.')[:-1])
            class_ = import_.split('.')[-1]
            if module not in imports_dict :
                imports_dict[module] = [class_]
            else:
                if class_ not in imports_dict[module]:
                    imports_dict[module].append(class_)

        return imports_dict


def blocks_to_script(blocks, prefix: str, imports):
    """ Set Blocks to script. """
    blocks_str = ""
    for iblock, block in enumerate(blocks):
        block_script = block._to_script(prefix)
        imports.extend(block_script.imports)
        if block_script.before_declaration is not None:
            blocks_str += f"{block_script.before_declaration}\n"
        blocks_str += f'{prefix}block_{iblock} = {block_script.declaration}\n'
    blocks_str += f"{prefix}blocks = [{', '.join([prefix + 'block_' + str(i) for i in range(len(blocks))])}]\n"
    return blocks_str


def nonblock_variables_to_script(nonblock_variables, prefix, imports, imports_as_is):
    """ Set Non Block Variables to script. """
    nbvs_str = ""
    for nbv_index, nbv in enumerate(nonblock_variables):
        nbv_script = nbv._to_script()
        imports.extend(nbv_script.imports)
        imports_as_is.extend(nbv_script.imports_as_is)
        nbvs_str += f"{prefix}variable_{nbv_index} = {nbv_script.declaration}\n"
    return nbvs_str


def update_imports(input_type, input_name):
    """ Update imports based on the schema of the input. """
    schema = get_schema(annotation=input_type, attribute=SchemaAttribute(input_name))
    return schema.get_import_names(import_names=[])


def generate_input_string(signature: str, sequence_length: int = 1):
    """ Generate input as string. """
    if sequence_length > 1:
        instances = [f"\n\t{signature}" for _ in range(sequence_length)]
        return f"[{','.join(instances)}\n]"
    return f"{signature}"


def is_object_sequence(sequence):
    """ Checks if a sequence contains serializable objects. """
    return all(isinstance(element, SerializableObject) for element in sequence)


def process_value(value):
    """ Processes a value based on its content. """
    if not value:
        return repr(value)

    sequence_length = len(value)

    if is_object_sequence(value):
        object_class = value[0].__class__.__name__
        signature = f"{object_class}('Set your arguments here')"
    elif is_file_or_file_sequence(value):
        object_class = value[0].__class__.__name__
        signature = f"{object_class}.from_file('Set your filepath here')"
    else:
        return repr(value)

    return generate_input_string(signature, sequence_length)
