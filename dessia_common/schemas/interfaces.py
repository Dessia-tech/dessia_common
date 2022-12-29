"""
Type Checking interfaces for schemas
"""
import typing as tp


class PropertySchema(tp.TypedDict):
    title: str
    editable: bool
    description: str
    python_typing: str
    type: str


class TupleSchema(PropertySchema):
    additionalItems: bool
    items: tp.Any

