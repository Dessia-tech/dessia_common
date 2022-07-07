from typing import List


class ToScriptElement:
    before_declaration: str = None
    declaration: str = None
    imports_as_is: List[str] = None
    imports: List[str] = None

    def __init__(self, declaration: str, before_declaration: str = None,
                 imports: List[str] = None, imports_as_is: List[str] = None):
        self.before_declaration = before_declaration
        self.declaration = declaration
        self.imports = imports
        self.imports_as_is = imports_as_is

