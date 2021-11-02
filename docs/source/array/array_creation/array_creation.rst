Array Constructors
##################

LibRapid arrays can be made in a wide variety of ways, allowing
greater functionality and flexibility when using the library.

Some of the constructors create entirely new data, where nothing
is referencing it, while others create arrays that reference the
data of another, so changing a value in one will also change the
value in the other.

When a new array is created with entirely new data, the memory
itself is contiguous (i.e. in a single block) in memory, meaning
it's incredibly fast to access and do calculations with.

Any function which takes a librapid `ndarray` object can also
take a python list, tuple or scalar, which will be automatically
cast to the ndarray type. This makes your code more concise and
accelerates development.

.. Hint::
	If you want to use an array that isn't contiguous (e.g. it
	was reshaped or is part of a larger array), the ``clone()``
	function will copy the data, but make it optimal in memory,
	improving performance where applicable.

.. toctree::
	:maxdepth: 2
	:glob:

	default_constructor
	extent_constructor
	from_array
	from_data
	val_like
	linear
	range
