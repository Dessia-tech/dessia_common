========
Advanced
========

#################
Implemented Types
#################

Custom Class
############
Subclass
***********
*Python type* :

``Subclass[Type]``

*TypeChecks* :

Any object which class inherits from *Type*.
*Type* must inherit from DessiaObject.

*Minimal jsonschema*:

.. code-block:: python

    {
    'type': 'object',
    'subclass_of': 'package_name.module_name.Type'
    }


Union
********
*Python type* :

``Union[Type1, Type2, ..., TypeN]``

*TypeChecks* :

Any object which class is one from *Type1*, *Type2*, *TypeN*.
These types must inherits from DessiaObject.

*Minimal jsonschema*:

.. code-block:: python

    {
    'type': 'object',
    'classes': ['package_name.module_name.Type1',
                'package_name.module_name.Type2',
                ...,
                'package_name.module_name.TypeN']
    }


Standalone Object
********************
*Python type* :

``Type``

*TypeChecks* :

Any object which class is *Type*.
*Type* must inherit from DessiaObject & define _standalone_in_db_ to ``True``

*Minimal jsonschema*:

.. code-block:: python

    {
    'type': 'object',
    'classes': ['package_name.module_name.Type']
    }


Embedded Object
******************
*Python type* :

``Type``

*TypeChecks* :

Any object which class is *Type*.
*Type* must inherit from DessiaObject & define _standalone_in_db_ to ``False``

*Minimal jsonschema*:

.. code-block:: python

    {
    'type': 'object',
    'classes': ['package_name.module_name.Type']
    }


Static Dict
**************
*Python type* :

.. code-block:: python

    class StaticDict(TypedDict):
      key1: BuiltinType (float)
      other_key: BuiltinType (str)
      ...
      nth_key: BuiltinType (bool)

*TypeChecks* :

A dictionnary that has same static structure than defined in StaticDict.
All value types must be builtins (not necessary the same)

*Minimal jsonschema*:

.. code-block:: python

    {
    'type': 'object',
    'properties': {...}
    }



Dynamic Dict
#############
*Python type* :

``Dict[str, Type2]``

*TypeChecks* :

non-defined-length dict of key that has type *Type1* (only string for now),
and values have type *Type2*.

*Minimal jsonschema*:

.. code-block:: python

    {
        'type': 'object',
        'patternProperties': {'.*': {'type': 'Type2'}}
    }



Heterogenous Sequence
######################
*Python type* :

``Tuple[Type1, Type2, ..., TypeN]``

*TypeChecks* :

n-lengthed sequence of specified types at specified positions
(equivalent of tuple)

*Minimal jsonschema*:

.. code-block:: python

    {
        'additionalItems': False,
        'type': 'array',
        'items': [
            { jsonschema of Type1 },
            { jsonschema of Type2 },
            ...,
            { jsonschema of TypeN }
        ]
    }



Homogeneous Sequence:
#####################
*Python type* :

``List[Type]``

*TypeChecks* :

non-defined-length dict of objects of type *type* (equivalent of list)

*Minimal jsonschema*:

.. code-block:: python

    {
        'type': 'array',
        'items': { jsonschema of Type }
    }



Builtin:
########
*Python type* :

``float | int | str | bool``

*Minimal jsonschema*:

.. code-block:: python

    {'type': 'builtin equivalence'}


Builtin equivalences are :

.. code-block:: python

    int : 'number'
    float: 'number'
    str: 'string'
    bool: 'boolean'
