Getting Started
***************


Introduction
============
DessiaCommon is Dessia's base package, providing generic models and tools to manipulate MBSEs and display them on DessIA's platform.
It is composed of three modules : core, workflow and vectored_objects.

Installation
============
Dessia Common package requires Python3.7.

For Windows user, we recommand using Anaconda Distribution.

https://www.anaconda.com/distribution/

Build your first bot
====================

Along this documentation, we will design a bot for a bearing architecture as a tutorial. Bearing models are found in DessIA's open-source MechanicalComponent library
In order to get started with DessIA's platform and build your bot, you first need to create a basic class inheriting from our base class DessiaObject.

dessia_common API Documentation :

:doc: `../core.rst`

.. autoclass:: dessia_common.core.DessiaObject

By inheriting DessiaObject, you actually "sign a contract" 



