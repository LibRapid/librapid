Array Constructors
##################

LibRapid arrays can be made in a wide variety of ways, allowing
greater functionality and flexibility when using the library.
Most constructors create an entirely new Array instance with its
own memory, however some constructors actually reference the
internal memory of another Array instance. This can be incredibly
useful in some situations, but can lead to  issues if you're not
aware that a given constructor operates like this. Make sure to
read the documentation for a given constructor to see how it
operates.

To make the code easier to use, any function which takes an Array
instance as an argument (in both C++ and Python) can also take a
list-like object or number, which will automatically be converted
into an Array.

For example, `librapid.add([1, 2, 3], [4, 5, 6])` will return an
Array containing `[5, 7, 9]`.

.. Hint::
	If you want to use an array that isn't contiguous (e.g. it
	was reshaped or is part of a larger array), the ``clone()``
	function will copy the data and make it optimal in memory,
	improving performance where applicable.

.. toctree::
	:maxdepth: 1
	:glob:

	array_constuctor_default
	array_constuctor_extent
	array_constuctor_array
	array_constuctor_data
	array_constuctor_like
	array_constuctor_linear
	array_constuctor_range
