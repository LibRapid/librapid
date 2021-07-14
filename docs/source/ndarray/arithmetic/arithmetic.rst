Array Arithmetic
################

Basic arithmetic is a key part of any array library, and
these operations are often used extensively throughout
programs. For this reason, a lot of work has been done to
optimise the unerlying routines that support the arithmetic
operations to make them as fast as possible. To make them
even faster, LibRapid may use multiple threads to perform
the calculations if they are available.

In both the Python and C++ libraries, the arithmetic
operators ``+ - * /`` are all overloaded, allowing you to
do things such as ``(a + b) / c``, though the following
methods are also implemented:

1. add(a, b)
2. sub(a, b)
3. mul(a, b)
4. div(a, b)

How arithmetic operations work
------------------------------

Arithmetic operations on arrays are all performed
element-wise, meaning the arrays must conform to specific
rules regarding their extents (see :ref:`array_shapes`)

For example, adding two arrays works as follows:

.. code-block:: python

	[1 2 3] + [4 5 6] = [5 7 9]
	# Because 1 + 4 = 5
	#     and 2 + 5 = 7
	#     and 3 + 6 = 9

.. array_shapes:

Array extents and formats
-------------------------

Librapid provides differnent methods for adding two arrays,
which vary based on the extents of the input arrays.

A full list of the arithmetic types is below:

+------------------+-------------------------------+-----------------------------------+
| Name             | Requirement                   | Example                           |
+==================+===============================+===================================+
| Exact match      | The arrays are identical      | .. code-block:: python            |
|                  | in their extents              |                                   |
|                  |                               |     [[1 2]  + [[5 6]  = [[ 6  8]  |
|                  |                               |      [3 4]]    [7 8]]    [10 12]] |
+------------------+-------------------------------+-----------------------------------+
| Outer dimensions | The arrays have the same      | .. code-block:: python            |
| match            | extent, but have leading      |                                   |
|                  | or trailing 1s                |     [1 2] + [[[3 4]]] = [4 6]     |
|                  |                               |                                   |
|                  |                               |     [[[1 2]]] + [3 4] = [[[4 6]]] |
+------------------+-------------------------------+-----------------------------------+
| Single value     | One of the addends is an      | .. code-block:: python            |
| array            | array with only a single      |                                   |
|                  | value                         |     [1 2 3] + [10] => [11 12 13]  |
|                  |                               |                                   |
|                  |                               |     [10] + [1 2 3] => [11 12 13]  |
+------------------+-------------------------------+-----------------------------------+
| Row-by-row       | The dimensions ``[0 .. n-1]`` | .. code-block:: python            |
|                  | of one array match the        |                                   |
|                  | dimensions of the other       |     [[1 2]  + [5 6] = [[ 6  8]    |
|                  |                               |      [3 4]]            [ 8 10]]   |
+------------------+-------------------------------+-----------------------------------+
| Grid             | Array with extent ``[... 1]`` | .. code-block:: python            |
|                  | and array with extent         |                                   |
|                  | ``[1 ...]``                   |     [1 2] + [[3]  = [[4 5]        |
|                  |                               |              [4]]    [5 6]]       |
+------------------+-------------------------------+-----------------------------------+
| Column-by-column | The dimensions ``[1 .. n]``   | .. code-block:: python            |
|                  | of one array match the        |                                   |
|                  | dimensions ``[0 .. m-1]`` of  |     [[1 2]  + [[5]  = [[ 6  7]    |
|                  | the other                     |      [3 4]]    [6]]    [ 9 10]]   |
+------------------+-------------------------------+-----------------------------------+

.. Attention::

	In the Python library, the ``true_div`` function is
	overloaded, not the standard ``div`` (integer division)
	function. To perform integer division, please use
	``floor(a / b)``

.. toctree::
	:maxdepth: 2
	:glob:

	add
	sub
	mul
	div
	array_addition
	array_subtraction
	array_multiplication
	array_division
	negate
