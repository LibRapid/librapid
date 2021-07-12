ndarray(shape)
###############

.. cpp:function:: librapid::basic_ndarray::basic_ndarray(const librapid::basic_extent &size)
.. cpp:function:: librapid::basic_ndarray::basic_ndarray(const std::initializer_list &size)

Create a new array from a given extent.

The array created will have the same number of dimensions
as the number of elements passed in the extent object. For
example, passing in ``extent(2, 3)`` will create a 2x3
matrix.

Parameters
----------

size: librapid::extent
    The dimensions of the array
