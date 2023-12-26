"""
Type Checking interfaces for schemas.

The aim of this module is to provide an "API documentation" or at least to state the body of schemas
in order to define a language understandable by both frontend and dessia_common.
It heavily relies on data typing and propose ways to type function return types, for example.
We could test that dessia_common is able to write structures validating these schemas
and ensure that frontend is able to read them.
By keeping them both side, it would decrease the risk of regression issues

This is mostly experimental and might require higher versions of python in order to work properly.
As an example, python >= 3.11 could provide a way to define required or not required attribute
in generated dictionaries.

At the very moment these lines are written (25 May 2023), Pycharm is complaining over dictionary specification.
The base dictionary generated for a Property validates its schema. However, when "extending" it to OptionalProperty,
for instance, it complains about type casting, and it seems that no solutions are currently available.
"""
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


class OptionalPropertySchema(PropertySchema):
    """ Typing for Optional properties. Experimental. """

    default_value: tp.Any


class TupleSchema(PropertySchema):
    """ Typing for Tuple Schemas. Experimental. """

    additionalItems: bool
    items: tp.List[PropertySchema]
