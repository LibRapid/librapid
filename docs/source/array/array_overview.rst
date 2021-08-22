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

The LibRapid Array can store arrays of any dimension, though the default
limit is 32 dimensions.

.. WARNING::
	While it is possible to change the maximum number of dimensions an array
	can contain, it is not recommended due to the extraordinary amount of
	memory that would be required to store such an array.

Why use an Array?
===================

The LibRapid Array class implements extremely optimized and efficent algorithms
for many mathematical operations, such as element-wise arithmetic, dot-products,
transpositions and more.

Because of this optimization, they can be used in high-intensity situations, such
as in the LibRapid neural network library, without compromising the speed of the
program or the range of functions available.

Additionally, when using LibRapid in C++, it is incredibly easy to manipulate the
Array type to fit you needs, as it is fully templated and works with a wide range
of datatypes, with many functions supporting cross-datatype operations.

How are the arrays stored?
==========================

The underlying memory of each array is stored in a contiguous memory block. To access
the different elements of an array, they also store the dimensions of the array, and
a set of strides which specify how far through memory one must move to increment by
one value in a given axis.

The fact that the arrays are stored in this way means that many functions can be
accelerated dramatically by reducing the amount of data that must be transferred.
For example, to transpose an array, the stride and extent are simply reversed.

Documentation
=============

.. toctree::
	:maxdepth: 2
	:glob:

	array_creation/array_creation
	arithmetic/arithmetic
	manipulation/manipulation.rst
