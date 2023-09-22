Glossary
========

Block
-----

A unit part of a workflow that can perform unit tasks such as object instantiation or method execution,...

Bot
---

Digitalization of a process including the writing of generic knowledge and a
workflow to describe the Know-How.

Deserialization
-------------

Writing a web compliant object (str) into a python object (in our case).
A first stage is a standard transformation from a JSON (string) to a dictionary and python's builtins.
Second stage is DessiaObject's ``dict_to_object`` method that recursively transforms a dictionary into an object.
This enables communication between python and our platform.

DessiaObject
------------

Framework base class that provides utilities for the use user classes on platform.

PhysicalObject
--------------

An extension of DessiaObject that denotes an object that can be displayed in 3D.

Platform
--------

Dessia's software solution based on a Web interface and a backend that emulates
Python.

Pipe
----

Links workflow blocks in order to pass data between them.

Schema
------

A schema is a web compliant represetation of a class or method/function structure.
It describes what attributes of which types are need to create an instance or run a function,
or which ones are constitutive of the class structure.

SDK
---

Set of Python libraries allowing to write in low-code the Know-That bricks
(generic knowledge).

Serialization
-------------

Writing a python (in our case) object in a web compliant language (str).
A first stage is DessiaObject's ``to_dict`` method that recursively transforms an object into its dictionary represention.
Second stage is a standard transformation from a dictionary and python's builtins into a string.
This enables communication between python and our platform.

Workflow
--------

A class that enables user to automate engineering process.
It is composed of blocks and pipes and can have cascading workflows in it.

