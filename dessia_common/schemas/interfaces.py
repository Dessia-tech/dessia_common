"""
Type Checking interfaces for schemas
"""
import typing as tp


T = tp.TypeVar("T")

Annotations = tp.Dict[str, tp.Type[T]]


class PropertySchema(tp.TypedDict):
    title: str
    editable: bool
    description: str
    python_typing: str
    type: str


class TupleSchema(PropertySchema):
    additionalItems: bool
    items: tp.List[PropertySchema]
