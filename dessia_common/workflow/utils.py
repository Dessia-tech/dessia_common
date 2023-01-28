#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utilities for workflows. """

from typing import List, Dict


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
