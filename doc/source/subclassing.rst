====================
Using Dessia Classes
====================

Base class: DessiaObject
########################

Dessia SDK offers several classes to inherit. These classes offer some capabilities usefull for Dessia platform.

The base class is dessia_common.DessiaObject. It is the most generic, it offers:

* tranforming the object to a dict and vice-versa (serialization/deserialization)
* various exports (json, XLSX, markdown)

For example, to create a class to represent some specifications:


.. code-block:: python

    from dessia_common.typings import Power, Distance
    from dessia_common import DessiaObject

    class CustomSpecifications(DessiaObect):
        def __init__(self, power_at_input:Power, power_at_output:Power, max_length:Distance, name:str=''):
            self.power_at_input = power_at_input
            self.power_at_output = power_at_output
            self.max_length = max_length
            DessiaObject.__init__(self, name=name)

This will define a data structure with 3 properties which are physical quantities (2 power in Watts and one distance in meters)

Physical Objects
################

A dessia Object does not have a CAD representation. The physical object adds this feature by defining a custom volmldr primitives to define CAD.

Let's define a machine class with some physical parameters, and use them to define and generate the CAD:

.. code-block:: python

    from dessia_common.typings import Power, Distance
    from dessia_common import PhysicalObject
    import volmdlr as vm
    from volmdlr.primitives3d import Cylinder

    class Machine(DessiaObect):
        def __init__(self, power:Power, diameter:Distance, length:Distance, name:str=''):
            self.power = power
            self.diameter = diameter
            self.length = length
            PhysicalObject.__init__(self, name=name)

        def volmdlr_primitives(self):
            return [Cylinder(vm.O3D, vm.X3D, 0.5*self.diameter, self.length)]

See volmdlr documentation to see all possibilities of volumic primitives (https://documentation.dessia.tech/volmdlr/)

Generators
##########

Dessia methodology is focused on enabling the generation of a lot of solutions

.. autoclass:: dessia_common.generation.DecisionTreeGenerator
    :members:

.. autoclass:: dessia_common.generation.RegularDecisionTreeGenerator
    :members:

