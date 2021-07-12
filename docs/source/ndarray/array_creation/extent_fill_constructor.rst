ndarray(shape, fill)
#####################

.. cpp:function:: librapid::basic_ndarray::basic_ndarray(const librapid::basic_extent &size, T val)
.. cpp:function:: librapid::basic_ndarray::basic_ndarray(const std::initializer_list &size, T val)

Creates a new array from a given shape, and fill it with a value.

For example, creating an array from ``librapid::extent({3, 4}), 5.``
will create the following array:

.. code-block:: python

    [[5. 5. 5. 5.]
     [5. 5. 5. 5.]
     [5. 5. 5. 5.]]

Parameters
----------

size: librapid::extent
    The dimensions of the array
val: Any arithmetic type
    The fill value
