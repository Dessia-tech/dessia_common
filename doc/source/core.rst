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

