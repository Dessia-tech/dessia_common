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

Your python package architecture is then created with provided information


For Linux user
==============

You can use open source package by running following shell command :
``pip3 install dessia_common``



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

