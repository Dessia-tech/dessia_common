Core
****


Dessia Object
=============

DessiaObject is the base on which models that have to go on a dessia platform should inherit from that class.

It allows python objects to have the following capabilities:
 * serialization/deserialization to travel in json on the web, and save & load to file
 * various exports to json & xlsx

.. autoclass:: dessia_common.DessiaObject
   :members:

Physical Object
===============

*added in version 0.7.0*

A PhysicalObject is a DessiaObject that has a physical representation. It should implement a custom method volmdlr_primitives.
It then enables to have:
- CAD display with .babylonjs method
- exports to stl and step format files

.. autoclass:: dessia_common.PhysicalObject
   :members:

Dessia Filter
===============

A Dessia Fitler is defined as an attribute name of a DessiaObject (e.g. 'mass'), a comparison operator (e.g. '>=' or 'le')
and a float value (e.g. 15.).
Its application on a list of DessiaObjects is the list of all contained DessiaObjects that satisfy the condition imposed by the current DessiaFilter.
Comparison operator can be one of:
- greater than: '>=', 'gte', 'ge' ;
- greater: '>', 'gt' ;
- lower than: '<=', 'lte', 'le' ;
- lower: '<', 'lt' ;
- equal: '==', 'eq' ;
- different: '!=', 'ne'

.. autoclass:: dessia_common.DessiaFilter
   :members:

