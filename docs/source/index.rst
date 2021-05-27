======================
LibRapid Documentation
======================

LibRapid -- a fast, general purpose math and neural network library for C++ and Python

.. important::
	To use LibRapid in Python, a modern C++ compiler may be needed to compile the code
	during installation. To use LibRapid in a C++ program, a modern C++ compiler will
	definitely be needed.

The Aim
=======

LibRapid aims to provide fast and intuitive interface to mathematical functions and
types, such as multi-dimensional arrays and neural networks.

LibRapid hopes to provide a functionally and interactively consistant interface in
both C++ and Python, allowing high level code to run faster and more efficiently,
while also ensuring low-level programs in C++ can be optimized in the same way.

.. Tip::

	LibRapid is a header-only library, meaning there are no binaries to be built!

Licencing
=========

LibRapid is produced under the Boost Software License, so you are free to ``use,
reproduce, display, distribute, execute and transmit`` the software, though this is
subject to some conditions, which can be found in full here: `LibRapid License`_

.. _LibRapid License: https://github.com/Pencilcaseman/librapid/blob/master/LICENSE

Installing the Package
======================

To install the Python package, simply open a command line window and type

.. code-block:: shell

	pip install librapid

To install the library for C++ useage simply download the code and either copy the
files to your program directory, or save them somewhere memorable (such as
``C:\opt\librapid``) and add them to the include direcotires of your project.

The simplest way to download the code is via Git, using the command

.. code-block:: shell

	git clone https://github.com/Pencilcaseman/librapid.git

Using LibRapid
==============

LibRapid is intended to be incredibly easy to use in both C++ and Python. To include
the library in one of your projects, simply do the following:


.. code-block:: Python
	:caption: For Python programs

	import librapid

.. code-block:: C++
	:caption: For C++ programs

	#include <librapid.hpp>


Contents
========

.. toctree::
	:maxdepth: 2
	:glob:

	ndarray_api

Test
====

.. doxy:c:: config.hpp::a_random_function
    :children:

Indices and tables
******************

* :ref:`search`
