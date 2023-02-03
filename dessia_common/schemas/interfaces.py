""" Type Checking interfaces for schemas. """
import typing as tp


T = tp.TypeVar("T")

Annotations = tp.Dict[str, tp.Type[T]]


class PropertySchema(tp.TypedDict):
    """ Typing for properties. Experimental. """

    title: str
    editable: bool
    description: str
    python_typing: str
    type: str


class TupleSchema(PropertySchema):
    """ Typing for Tuple Schemas. Experimental. """

    additionalItems: bool
    items: tp.List[PropertySchema]
