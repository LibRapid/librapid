================
LibRapid Array
================

What is an Array?
===================

An Array is a multi-dimensional, homogeneous collection of values which
can be operated on in a variety of ways.

The simplest type of Array is a vector, which is a list of values.
Another simple type of Array is a matrix, which is a grid of values.

.. code-block:: C++

	// This is a vector with 3 elements
	[1 2 3]
	
	// This is a 2x3 matrix
	[[1 2 3]
	 [4 5 6]]

	// This is a 1x1x1x1 array
	[[[[123]]]]

The LibRapid Array type can store arrays with any number of dimensions, though the
default limit is 32D.

.. WARNING::
	While it is possible to change the maximum number of dimensions an array
	can contain, it is not recommended due to the extraordinary amount of
	memory that would be required to store such an array.

Why use an Array?
===================

The LibRapid Array class implements extremely optimized and efficent algorithms
for many mathematical operations, such as element-wise arithmetic, matrix calculations,
reduction operations and more.

As a result of these optimisations, the Array class can be used to accelerate
calculations and reduce running time. This can be incredibly important for certain
programs where calculations need to be run many times, over and over again. Saving
a few milliseconds here and there can add up to hours being saved in the long run.

How are the arrays stored?
==========================

The underlying memory of each array is stored in a contiguous memory block. To access
the different elements of an array, they also store the dimensions of the array and
a set of strides which specify how far through memory to move to increment by one in
a given axis.

The fact that the arrays are stored in this way means that many functions can be
accelerated dramatically by reducing the amount of data that must be transferred.
For example, to transpose an array, the stride and extent are simply reversed.

.. Hint::
	For more detailed information the low-level details of the Array implementation and
	optimisations, take a look at the Optimisation page in the documentation.

Documentation
=============

.. toctree::
	:hidden:
	:maxdepth: 2
	:glob:

	array_properties/array_properties
	array_constructors/array_constructors
	array_arithmetic/array_arithmetic
	array_general_utilities/array_general_utilities
	array_manipulation/array_manipulation

.. panels::
	Properties

	+++

	.. link-button:: array_properties/array_properties
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

	---

	Constructors

	+++

	.. link-button:: array_constructors/array_constructors
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

	---

	General Utilities

	+++

	.. link-button:: array_general_utilities/array_general_utilities
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

	---

	Arithmetic

	+++

	.. link-button:: array_arithmetic/array_arithmetic
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

	---

	Manipulation

	+++

	.. link-button:: array_manipulation/array_manipulation
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link
