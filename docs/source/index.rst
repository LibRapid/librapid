.. image:: ../../branding/logo_transparent_trimmed.png
   :width: 800

.. image:: https://github.com/pencilcaseman/librapid/actions/workflows/wheels.yaml/badge.svg
	:target: https://github.com/Pencilcaseman/librapid/actions/workflows/wheels.yaml
.. image:: https://readthedocs.org/projects/librapid/badge/?version=latest
	:target: https://librapid.readthedocs.io/en/latest/?badge=latest
.. image:: https://badge.fury.io/py/librapid.svg
	:target: https://pypi.python.org/pypi/librapid/
.. image:: https://img.shields.io/pypi/l/librapid.svg
	:target: https://pypi.python.org/pypi/librapid/
.. image:: https://img.shields.io/pypi/pyversions/librapid.svg
	:target: https://pypi.python.org/pypi/librapid/
.. image:: https://img.shields.io/pypi/dm/librapid.svg
	:target: https://pypi.python.org/pypi/librapid/
.. image:: https://img.shields.io/discord/848914274105557043
	:target: https://discord.gg/cGxTFTgCAC


The Aim
=======

LibRapid aims to provide fast and intuitive interface to mathematical functions and
types, such as multi-dimensional arrays and neural networks.

LibRapid hopes to provide a functionally and interactively consistant interface in
both C++ and Python, allowing high level code to run faster and more efficiently,
while also ensuring low-level programs in C++ can be optimized in the same way.

What LibRapid is
----------------

LibRapid is a lightweight alternative to other popular python libraries, intended
to be used in both C++ and Python. It also supports NVIDIA CUDA (even in Python), so
you can run array calculations orders of magnitude more quickly than on the CPU alone.
It is designed to be as easy to use as possible while still supporting advanced
functionality and high performance.

What LibRapid is NOT
--------------------

A replacement for the well-established Python and C++ libraries that already
exist. It can be used in a wide range of applications, but there are no guarantees
that all functions perform exactly the correct calculations in all situations,
though increasing numbers of tests are being implemented, and any bugs discovered
are fixed as quickly as possible.

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

.. important::
	To use LibRapid in Python, a modern C++ compiler may be needed to compile the code
	during installation. To use LibRapid in a C++ program, a modern C++ compiler will
	definitely be needed.

LibRapid is intended to be incredibly easy to use in both C++ and Python. To include
the library in one of your projects, simply do the following:


.. code-block:: Python
	:caption: For Python programs

	import librapid

.. code-block:: C++
	:caption: For C++ programs

	#include <librapid/librapid.hpp>

Using the Documentation
=======================

The documentation outlines the explicit declerations for the C++ functions themselves,
which may be difficult to understand for many users. Luckily, however, only a small
portion of the function definitions is actually relevant to most users.

For example, take the function definition below:

.. cpp:function:: template<typename T> inline double librapid::test_function(const test_struct &thing) const

It may appear complicated at first, but there are only a few key things to look out
for. First, the name of the function (``test_function``) will be the same in the
Python and C++ libraries, except in Python, the function will be accessed as
``librapid.test_function``. Next, the arguments to the function are important to know,
and should be self-explanatory. In C++, the type can be seen in the function definition,
and passing the same type should give error-free code. In Python, however, things can be
slightly different -- often, Python values can be cast into equivelent C++ types, though
the types are not always clear. Below is a list of common datatypes that you may want to
pass to functions:

+---------------+------------------------+-----------------+
| Type of value | C++ Datatype           | Python Datatype |
+===============+========================+=================+
| List          | ``std::vector<...>``   | ``list, tuple`` |
+---------------+------------------------+-----------------+
| Integer       | ``int64_t, int, long`` | ``int``         |
+---------------+------------------------+-----------------+
| Dictionary    | ``std::map<a, b>``     | ``dict``        |
+---------------+------------------------+-----------------+

Contents
========

.. toctree::
	:maxdepth: 2
	:glob:

	array/array_overview
