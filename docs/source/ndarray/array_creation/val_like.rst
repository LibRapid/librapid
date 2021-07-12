zeros, ones, random _like(ndarray)
###################

.. cpp:function:: librapid::basic_ndarray librapid::zeros_like(const librapid::basic_ndarray &data)
.. cpp:function:: librapid::basic_ndarray librapid::ones_like(const librapid::basic_ndarray &data)
.. cpp:function:: librapid::basic_ndarray librapid::random_like(const librapid::basic_ndarray &data)

Create and return a new array with the same shape
as the provided array, but filled entirely with zeros,
ones or random values in a given range

The datatype of the returned array will be the same
as the type of the input array.

.. Attention::

	When using the ``librapid::random_like`` function
    please note that the ``librapid::math::random``
	function returns values in the range ``[min, max]``
	for integer values, though in the range ``[min, max)``
	for floating point values (i.e. floating point values
	will never exceed the value of ``max``)

Parameters
----------

arr: basic_ndarray
	The array to base the size off
min = 0: any arithmetic type
	The minimum random value (only when using ``random_like``)
max = 1: any arithmetic type
	The maximum random value (only when using ``random_like``)

Returns
-------

result: basic_ndarray
	A new array filled with random values in the
	specified range
