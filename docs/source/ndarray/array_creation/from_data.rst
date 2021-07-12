ndarray(data)
#############

.. cpp:function:: librapid::basic_ndarray::from_data(const dtype &data)
.. cpp:function:: librapid::basic_ndarray::from_data(const std::vector &data)
.. cpp:function:: librapid::basic_ndarray::from_data(const std::vector<std::vector> &data)

Create a new array from the provided data. The function supports
creation from a scalar value, as well as vectors, matrices and
higher dimensional data.

For arrays with dimensions greater than or equal to 2 (i.e. matrices
and above) the sub-arrays *must* be the same size, otherwise an error
will be thrown.

Examples
--------

.. code-block:: C++
    :caption: C++ Example

    auto my_vec = librapid::ndarray::from_data(VEC{1, 2, 3, 4});
    std::cout << my_vec << "\n";
    // Prints: [1. 2. 3. 4.]

.. code-block:: Python
    :caption: Python Example

    my_vec = librapid.ndarray.from_data([1, 2, 3, 4])
    print(my_vec)
    # Prints: [1. 2. 3. 4.]

C++ Specific
------------

In C++, the input must be specifically denoted as ``std::vector<...>``
for vectors and arrays (this may not be needed when using MSVC). To
shorten this process, the typename ``VEC<...>`` is provided to allow
the following: ``librapid:: ... ::from_data(VEC<VEC<int>>{{1, 2}, {3, 4}})``.

Python Specific
---------------

Due to Python's dynamic typing features, a ``list`` or ``tuple`` can be
easily converted to an array -- there's no need to worry about any conversions.

Parameters
----------

data: scalar, vector, nested-vector
    The data for the array
