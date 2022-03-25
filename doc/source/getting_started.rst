===============
Getting Started
===============


Introduction
############
DessiaCommon is Dessia's base package, providing generic models and tools to manipulate MBSEs and display them on DessIA's platform.
It is composed of three main modules : core, workflow and vectored_objects.

Installation
############

Python
******

Dessia Common package requires Python3.7

For Windows user, we recommand using Anaconda Distribution.

https://www.anaconda.com/distribution/

Git
***

You may need to use git in order to manage your project versions and to share your work with your colleagues. In that extent please follow `git guidelines <https://docs.github.com/en/free-pro-team@latest/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line>`_ to setup you own repository.


Installing
**********

.. code-block:: console
    pip install dessia_common

On linux, pip may be named pip3 (pip for python3)


Dessia Bot Template
*******************

In order to create your first bot, Dessia provide a quick way to setup you project, via its small and simple `dessia_bot_template package <https://github.com/Dessia-tech/dessia_bot_template>`_.
You can downlad and unzip these files anywhere on your computer.

Afterward, run our wizard by running following shell commands (in AnacondaPrompt if you use Anaconda distribution) :

.. code-block:: console

    cd dessia_bot_template
    python3 quickstart.py

Your python package architecture is then created with provided information. This also installs lastest versions of dessia_common as well as volmldr, by default.

You can install your package by navigating to the folder you created for your bot 

.. code-block:: console

    cd [...]/my_first_bot
    python3 setup.py develop --user

This install the package for the used Python (here python3) environment.

You are now ready to write your first class and methods !



