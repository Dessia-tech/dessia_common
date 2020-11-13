===============
Getting Started
===============


Introduction
############
DessiaCommon is Dessia's base package, providing generic models and tools to manipulate MBSEs and display them on DessIA's platform.
It is composed of three modules : core, workflow and vectored_objects.

Installation
############

Python
******

Dessia Common package requires Python3.7.

For Windows user, we recommand using Anaconda Distribution.

https://www.anaconda.com/distribution/

Git
***

You may need to use git in order to manage your project versions and to share your work with your colleagues. In that extent please follow `git guidelines <https://docs.github.com/en/free-pro-team@latest/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line>`_ to setup you own repository. This is optionnal.

Let's say you want your project located under documents, initialize your git repository there. If you don't want to use git, just create your folder.

Documents/my_first_bot

Dessia Bot Template
*******************

In order to create your first bot, Dessia provide a quick way to setup you project, via its small and simple `dessia_bot_template package <https://github.com/Dessia-tech/dessia_bot_template>`_.
You can downlad and unzip these files anywhere on your computer.

Afterward, run our wizard by running following shell commands (in AnacondaPrompt if you use Anaconda distribution) :

``cd dessia_bot_template``

``python3 quickstart.py``

Your python package architecture is then created with provided information. This also installs lastest versions of dessia_common as well as volmldr, by default.

You are now ready to right you first class and methods !


Build your first bot
####################

Along this documentation, we will design a bot for a bearing architecture as a tutorial. Bearing models are found in DessIA's open-source MechanicalComponent library
In order to get started with DessIA's platform and build your bot, you first need to create a basic class inheriting from our base class DessiaObject.

.. dessia_common API reference Documentation :

.. :doc: `../core.rst`

.. autoclass:: dessia_common.core.DessiaObject

By inheriting DessiaObject, you actually "sign a contract" as DessiaObject alters base functionalities of Python.
By default, base equalities and hash computation are overwritten. Generic serialization/deserialization functions as well as jsonschema computation are defined.
On a more "engineering level", check for 2D/3D displays are mutualized, as are graph & parallel plots, for example.

The behavior of your DessiaObject inheriting class can be customized by changing the value of several class attributes such as ``_standalone_in_db``.

But, first, let's take our first step into coding.

Writing in Python requires to follow several strong naming convention and coding style that are defined in `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
This is not mandatory, but highly recommended. We will mention every small tidbits each time a new type of python object is encountered

As Python is an Object Oriented language, it enables us to defines classes that are concepts that regroupe variables and method.
This following line define a class that is named MyFirstClass :

``class MyFirstClass(DessiaObject):``

Here, PEP8 states that a class should be named with the CamelCase convention. In that extend, we will know for a fact that MyFirstClass and DessiaObject are class objects (and not functions, methods, instances, attributes,...)

What is the meaning of the whole line above ?

* ``class`` is a keyword that indicates we want to define a new class named after the following word : 
* ``MyFirstClass`` is the name of our new class.
* ``(DessiaObject)`` means that this class will inherit from DessiaObject's and use its attributes, and methods.
* ``:`` shows end of class definition.

As it stands, our class is not very useful. It needs a method that will enable us to create an instance of it.

``def __init__(self, my_attribute: float, name: str = ''):``

* ``def`` is a keyword that indicates we want to define a new function or method named after the following word :
* ``__init__`` is the method's name. In that case, it needs to be written as such, as it is imposed by Python. ``__init__`` is the method invoked when we try to instanciate an object.
* ``(self, ``. self is the first argument givent to the method. It is mandatory when writing any method and represent the instance that is being modified (or in our case, created). self knows every attribute of the instance.
* ``my_attribute`` is the name of the first attribute we want to set inside our class. We wille see in the next paragraph how to do so.
* ``: float`` is the type of my_attribute attribute. Normally, these are not mandatory in Python but we use them in order to build schemas of our objects. It enables us to create edition or creation forms, for example. In that extent, be sure to always type you functions and methods definition, as it is required by DessiaObject.
* ``name: str = ''):`` is our second attribute and end of the definition of our method. It is built exactly like the previous one (attribute name, attribute type) except we add a default value to name, in that example, an empty string.

From now, there  are only to steps that separate us from creating our first instance of MyFirstClass :
* Set my_attribute in ``MyFirstClass`` structure.
* Initialize DessiaObject in ``__init__`` of MyFirstClass as it inherits from it.

On the next line, by writing ``self.my_attribute = my_attribute``, we actually set it in the structure.
To complete ``__init__`` method, we intialize DessiaObject as such : 

``DessiaObject.__init__(self, name=name)``

Here is the full code for ``MyFirstClass`` at this state : 

.. code-block:: python

    class MyFirstClass(DessiaObject):
        def __init__(self, my_attribute: float, name: str = ''):
            self.my_attribute = my_attribute
            
            DessiaObject.__init__(self, name=name)




 An object is an instance of 

__eq__ and __hash__
*******************


``__eq__`` and ``__hash__`` rule how objects are behaving whenever we test if one is equal to one another.


By default, Python use the method from type object (python base object) that only check for strict equalities.

It means that ``==`` and ``is`` method are equivalent, and checks for a strict equality, on object adress in computer memory.


Overwriting ``__eq__`` enables us to redefine ``==`` so that it is based on data.

It is important if we want to store involved object in our database, setting its class attribute ``_standalone_in_db`` to True.

As a matter of fact, MongoDB needs an equality on data to function properly.

It also has a conceptual meaning in our *Object Oriented Engineering* vision as, for instance, two bearings that have exactly same dimensions are, physically speaking, the same object.


A custom ``__eq__`` method needs a relevant ``__hash__`` are the two are working together.


A hash is an integer value that is equivalent to an identifier. Two objects that are equal, on a data level, *must* share the same hash. It is a necessary but not sufficient condition, as two objects with the same hash might not be equal.

The contraposition is that have different hashes are not equal.


DessiaObject defines generic ``__eq__`` and ``__hash__`` functions that are based on following class attributes : 

* ``_non_data_eq_attributes (['name'])`` 
* ``_non_data_hash_attributes (['name'])`` 


Any attribute listed in these sequence (by default, just DessiaObject's name) aren't taken into account for equalities.


DessIA's Tidbits :
==================


* A class that is to be ``standalone_in_db`` must define a custom ``__eq__`` and ``__hash__``. Either DessiaObject's generic method or a user custom method.
  In that extent, the class definition might be as follows : 
  ``_standalone_in_db = True
    # Following is optionnal as this is default behavior :
    _eq_is_eq_data = True
    _non_data_eq_attributes = ['name', 'other_custom_attributes',...]
    _non_data_hash_attributes = ['name', 'other_custom_attributes',...]``

