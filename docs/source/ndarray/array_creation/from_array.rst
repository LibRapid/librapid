ndarray(ndarray)
################

.. cpp:function:: librapid::basic_ndarray::basic_ndarray(const librapid::basic_ndarray &arr)

Create a new array from an existing one, where the new array's data
is linked to the existing one's, so an update in one will update
the data in the other.

.. Attention::
	The shape and stride of the arrays will be the same, so a sub-optimal
	stride in the original array will also result in a sub-optimal stride
	in the new one. To mitigate this, try using ``.copy()`` instead.

Parameters
----------

arr: librapid::ndarray
    The array to reference