Getting started
===============

This package requires Python 3.8 or above. Please follow the instructions
below to install the package. Depending on your needs, you can choose between
two types of installation.

Installing dessia_common through pip (Windows and Linux users)
--------------------------------------------------------

To install the latest version of the package you need to run the following
command::

  pip install dessia_common
  # or
  pip3 install dessia_common

To install a specific version of the package you would issue the following
command::

  pip install dessia_common==0.1.0
  # or
  pip3 install dessia_common==0.1.0

Developer installation
----------------------

First, clone the package. Then, enter the newly created dessia_common repository. Finally, develop the setup.py file, and you are good to go ! ::

  git clone https://github.com/Dessia-tech/dessia_common.git

  cd dessia_common

  pip install dessia_common -e .

Requirements
------------

The installation of dessia_common requires the installation of other packages listed
in the file setup.py and in the table below. These libraries will be
automatically installed when you install dessia_common.

=============  ===============  ===========
Dependency     Minimum Version  Usage
=============  ===============  ===========
orjson         3.8.0            computation
networkx       latest           computation
numpy          latest           computation
scipy          latest           computation
pyDOE2         latest           computation
dectree        latest           computation
openpyxl       latest           computation
parameterized  latest           test
scikit-learn   1.2.0            computation
cma            latest           computation
docx           latest           computation
python-docx    latest           computation
matplotlib     latest           display
=============  ===============  ===========

Troubleshooting
---------------

If the installation is successful but your IDE don't recognize the package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case you may have several versions of Python installed on your
computer. Make sure the `pip` command points to the right Python version, or
that you have selected the desired Python version in your IDE.
You can force the installation of the package on a given Python version by
executing this command::

  python -m pip install dessia_common

You have to specify the Python version you are working with by replacing
`python` by the Python of your choice. For example, `python3`, `python3.8`,
`python3.9`, etc.
