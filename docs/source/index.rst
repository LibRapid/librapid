LibRapid
########

What is LibRapid?
------------------

LibRapid is a fast and efficient library for accelerating mathematical calculations
in Python and C++. LibRapid also provides an identical interface between C++ and Python,
allowing any program to be easily translated between languages.

The most fundamental component to LibRapid is it's `Array` object, capable of storing arrays
with up to 32 dimensions, using runtime-defined datatypes and running code on multiple threads
on the CPU, or in parallel on the GPU with CUDA. Every function is highly optimised for the
best possible performance, meaning your code will be faster as a result. Benchmarks show that
LibRapid is consistently faster than NumPy, and can, in some cases, exceed the performance of
Eigen (even in Python!)

Additionally, LibRapid supports a wide range of utility classes and functions, such as common
mathematical operations, a versatile Vector library, arbitrary-precision arithmetic, and more.
The pre-built Python wheels even come with OpenBLAS to further accelerate matrix calculations

Installing the Library
======================

To install the Python package, simply open a command line window and type

.. code-block:: shell

	pip install librapid


To use the library in C++, there are a variety of methods you can use. The preferred method
is to use ``CMake`` and ``FetchContent``, which will automatically clone, build and link the
library for you. To do this, simply add the following lines to your ``CMakeLists.txt`` file:

.. code-block:: cmake

	add_executable(MyApp myapp.cpp)

	include(FetchContent)
	FetchContent_Declare(librapid GIT_REPOSITORY https://github.com/librapid/librapid.git)
	FetchContent_MakeAvailable(librapid)

	target_link_libraries(MyApp librapid)


You can also clone the repository into your project (or add it as a submdoule) and add
the LibRapid subdirectory:

.. code-block:: shell

	git clone https://github.com/librapid/librapid.git --recursive

.. code-block:: cmake

	add_executable(MyApp myapp.cpp)

	add_subdirectory(librapid)

	target_link_libraries(MyApp librapid)


Advanced -- Installing LibRapid
===============================

.. important::
	To build LibRapid from source, a C++17 compatible compiler is required

If you want the best performance for Python, it is highly advised to build the library from source
as opposed to installing a pre-built version of it. This is because the pre-built binaries need to
function on a wide range of systems, and therefore cannot be as optimised as we would like. Luckily,
building the library from source is very simple:

1. Install the Python build requirements:
	- ``pip install -r requirements.txt``
2. (optional) Download/Build OpenBLAS and place the files somewhere LibRapid can find them
	- LibRapid will search for common install locations such as ``C:\opt\OpenBLAS``, though to be
	  entirely certain LibRapid will find the BLAS files, please place the build files in
	  ``src/librapid/openblas_install``
	- To download a platform-independent, optimised build of OpenBLAS for MacOS, Windows or Linux:
		- Go to https://github.com/librapid/librapid
		- Click on "Actions"
		- Click on the latest one to have passed (ensure it is labeled "Wheels")
		- Scroll down and select the OpenBLAS build for the desired platform (Note that the Ubuntu
		  build should work on any Linux install)
3. Option 1 -- Install source distribution (No OpenBLAS support unless installed on system)
	- Simply run ``pip install librapid --no-binary librapid``
4. Option 2 -- Install from source code
	- Clone the repository: ``git clone https://github.com/librapid/librapid.git``
	- Open a command line window in the LibRapid directory
	- Run ``pip install .``

.. hint::
	The OpenBLAS information applies to C++ as well, as CMake will search for OpenBLAS on your
	system as well as in the ``openblas_install`` directory!

Using LibRapid
==============

LibRapid is intended to be incredibly easy to use in both C++ and Python. To include
the library in one of your projects, simply do the following:


.. code-block:: python
	:caption: For Python programs

	import librapid

.. code-block:: cpp
	:caption: For C++ programs

	#include <librapid/librapid.hpp>

Using the Documentation
=======================

The documentation outlines the explicit declerations for the C++ functions themselves,
which may be difficult to understand for many users. Luckily, however, only a small
portion of the function definitions is actually relevant to most users.

For example, take the function definition below:

.. cpp:function:: double librapid::testFunction(const testStruct &thing) const

Using this function in C++ should be fairly self-explanatory, though using it in Python might
seem a little more tricky. The important things to look out for are:
- The name of the function -- this will be the same in Python and C++
- The arguments to the function -- the same arguments with the same names and default values will be used in Python and C++
- The return value -- The type of the value will be roughly the same in Python as it is in C++

.. attention::
	Some Python types need to be cast into C++ types. The type conversion is often obvious,
	but some types can be a little more confusing. Some commonly used ones are shown in the
	table below

+---------------+------------------------+-----------------+
| Type of value | C++ Datatype           | Python Datatype |
+===============+========================+=================+
| Decimal       | ``double, float``      | ``float``       |
+---------------+------------------------+-----------------+
| Integer       | ``int64_t, int, long`` | ``int``         |
+---------------+------------------------+-----------------+
| List          | ``std::vector<...>``   | ``list, tuple`` |
+---------------+------------------------+-----------------+
| Dictionary    | ``std::map<a, b>``     | ``dict``        |
+---------------+------------------------+-----------------+
| None-type     | ``void``               | ``None``        |
+---------------+------------------------+-----------------+

Contents
========

.. toctree::
	:hidden:
	:maxdepth: 2
	:glob:

	array/array_overview

.. panels::
	Array Overview

	+++

	.. link-button:: array/array_overview
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

	---

	Vector Overview

	+++

	.. link-button:: array/array_overview
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

Licencing
=========

LibRapid is produced under the MIT License, so you are free to use the library
how you like for personal and commercial purposes, though this is subject to
some conditions, which can be found in full here: `LibRapid License`_

.. _LibRapid License: https://github.com/Pencilcaseman/librapid/blob/master/LICENSE
