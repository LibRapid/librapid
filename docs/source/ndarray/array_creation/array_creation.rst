Array Constructors
##################

LibRapid arrays can be made in a wide variety of ways, allowing
greater functionality and flexibility when using the library.

Some of the constructors create entirely new data, where nothing
is referencing it, while others create arrays that reference the
data of another, so changing a value in one will also change the
value in the other.

.. toctree::
	:maxdepth: 2
	:glob:

	default_constructor
	extent_constructor
	extent_fill_constructor
	from_array
	from_data
	val_like
	linear
	range
