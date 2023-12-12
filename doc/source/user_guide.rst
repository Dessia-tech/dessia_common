User guide
==========

General description
-------------------

The dessia_common aims to provide a framework for engineering objects. this has two usages : 
- Provide engineering teams with utilities and best practices on how to model their structures informatically (as 'Object-Oriented Engineering')
- Provide base methods to make these structures able to communicate with web platforms.

dessia_common provides generic serialization process, generic 'schemas' descriptions of engineering structures, generic copy/equality computation, and so on...

Getting started
---------------

DessiaObject
^^^^^^^^^^^^

The framework offers several classes to inherit from.

The base class is DessiaObject, which is the most generic. Here is a quick overview of its capabilities:

* Write the object as a dict and vice-versa (serialization/deserialization),
* Read user code and generate classes and methods 'schemas',
* Provide object handling utilities as generic copy or equality between objects,

.. code-block:: python

    from dessia_common.typings import Power, Distance
    from dessia_common import DessiaObject

    class CustomSpecifications(DessiaObect):
        def __init__(self, power_at_input: Power, power_at_output: Power, max_length: Distance, name: str = ""):
            self.power_at_input = power_at_input
            self.power_at_output = power_at_output
            self.max_length = max_length

            DessiaObject.__init__(self, name=name)

DessiaObject is shipped with a set of class attributes that enables user to fine tune its behavior on the platform, but also in local use : 

.. autoclass:: dessia_common.core.DessiaObject


PhysicalObject
^^^^^^^^^^^^^^

PhysicalObject is basically a DessiaObject with a 3D view.
It inherits from all DessiaObject capabilities and extends it to add 3D properties.

.. autoclass:: dessia_common.core.PhysicalObject


Annotating your code
^^^^^^^^^^^^^^^^^^^^

As python is not a strongly type language, we are usually not used to have type hints playing a huge role in our codes.

However, as dessia_common is an engineering framework that handles generic object structures, typings are important for it to know what it is manipulating.

That is why, a substantial part of dessia_common is to provide types and functions around them for you to give the framework hints on how to handle things.

As an example, the platform allows you to create or edit object, as well as run functions, directly from the web app.
To do so, a form is dynamically created in order to provide the user with good inputs.

In that extent, how would a generic tool be able to provide the good inputs without any hints ? ie. does the code expect an int, a float, or a file for this specific input ?

That is where typings and annotations step in. Consider the following bit of code : 

.. code-block:: python

    class RoundedBeam(PhysicalObject):
        def __init__(self, diameter, length, name)
            self.diameter = diameter
            self.length = length
            
            PhysicalObject.__init__(self, name=name)


Without anymore information, the framework has no clue on what is expected in order to create an EmptyCylinder.
Annotating the ``__init__`` function would give it these necessary hints.

.. code-block:: python

    class RoundedBeam(PhysicalObject):
        def __init__(self, diameter: float, length: float, name: str = "")
            self.diameter = diameter
            self.length = length
            
            PhysicalObject.__init__(self, name=name)

In this example, this is pretty straightforward. We can obviously create more complex structures.


.. code-block:: python

    class Frame(PhysicalObject):
        def __init__(self, beams: List[Union[RoundedBeam, SquaredBeam]], connected: bool, name: str = "")
            self.beams = beams
            self.connected = connected
            
            PhysicalObject.__init__(self, name=name)


Annotations are not only useful for form generation. They are used everywhere in Dessia's applications.
They are also an incredible tool to use in order to well document the code.


Displays
--------

Displays are a key feature of dessia_common.
Your objects inheriting from DessiaObject can be displayed in the platform.
Therefore, it can be shipped with number of different views, that you can define in the code (3D, Markdown, 2D, Plots,...)

dessia_common provides helper features to help you define your views. By applying decorators to your methods, you'll be able to tell the framework that this function produces a display.
Display decorators can be found under dessia_common.decorators : 


.. code-block:: python

    from dessia_common.decorators import cad_view, markdown_view, plot_data_view


.. autofunction:: dessia_common.decorators.cad_view

.. autofunction:: dessia_common.decorators.markdown_view

.. autofunction:: dessia_common.decorators.plot_data_view

By using one of this decorator on a method of your class, you will tell dessia_common this method produces an object that is of display of given type.

.. code-block:: python

    from dessia_common.decorators import cad_view, markdown_view, plot_data_view
    from dessia_common import PhysicalObject

    class ClassWithCadDisplay(PhysicalObject):
        """ Dummy class to illustrate display features. """
        def __init__(self name: str = ""):

            PhysicalObject.__init__(self, name=name)

        @cad_view("CAD")
        def method_that_produces_a_3d(self):
            """ Generate a 3D display. """
            # Build your volmdlr code here, ex :
            volume = ExtrudedProfile(...)
            return [volume]

        @markdown_view("Markdown Report", load_by_default=True)
        def method_that_produces_a_md_report(self):
            """ Generate a markdown. """
            # Build your markdown here, ex :
            contents = "This is a *markdown report*"
            return contents

.. image:: display_selectors.jpg
    :alt: img: From precedent code, these selectors are available on platform



Return types of display-decorated methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

==============  =====================  =======================
Decorator       Display Type           Return Type
==============  =====================  =======================
cad_view        3D - Babylon           List[VolmdlrPrimitives]
plot_data_view  2D/Graphs - Plot Data  PlotDataObject
markdown_view   Markdown               str
==============  =====================  =======================


Exports
-------

Just like displays, your classes can define ways to be exported as various files (json, xlsx, zip,...).

Several "builtin" export are already available from the class you inherit from (DessiaObject, PhysicalObject).

DessiaObject can be exported by default as a .json, .xlsx, .zip or .docx.
PhysicalObject extends these export formats by adding .step, .html & .stl exports.

Definition of PhysicalObject export formats : 

.. code-block:: python

        def _export_formats(self) -> List[ExportFormat]:
        """ Return a list of objects describing how to call 3D exports. """
        formats = DessiaObject._export_formats(self)
        formats3d = [ExportFormat(selector="step", extension="step", method_name="to_step_stream", text=True),
                     ExportFormat(selector="stl", extension="stl", method_name="to_stl_stream", text=False),
                     ExportFormat(selector="html", extension="html", method_name="to_html_stream", text=True)]
        formats.extend(formats3d)
        return formats

You can add your own export formats by overriding the _export_format method. Doing so is as straightforward than what is done for PhysicalObject, above.

.. code-block:: python

    from dessia_common import PhysicalObject
    from dessia_common.exports import ExportFormat

    class ClassWithCustomExports(PhysicalObject):
        """ Dummy class to illustrate export usages. """
        def __init__(self name: str = ""):

            PhysicalObject.__init__(self, name=name)

        def to_custom_export_stream(self, stream: Union[StringFile, BinaryFile]):
            """ This is useful for platform usage. Platform uses streams to handle files. """"
            # Building stream

        def to_custom_export(self, filepath: str):
            """ This is useful for local usage. File gets generated at filepath location. """"
            # Building file
            if not filepath.endswith('.ext'):
            filepath += '.ext'
            print(f'Changing name to {filepath}')

            with open(filepath, 'wb') as file:
                self.to_custom_export_stream(file)

        def _export_formats(self) -> List[ExportFormat]:
            """ Extends (or not) default PhysicalObject exports to add custom ones. """
            formats = PhysicalObject._export_formats(self) # Erase this line if you wish to get rid of default exports.
            custom_formats = [ExportFormat(selector="Custom Selector", extension="ext", method_name="to_custom_export_stream", text=True)]
            formats.extend(custom_formats) # If defaults were kept
            return formats


.. autoclass:: dessia_common.exports.ExportFormat


By convention, methods that handle streams would be called ``to_x_stream`` and methods that handle files would be called ``to_x``



Q&A
---

No questions yet.
