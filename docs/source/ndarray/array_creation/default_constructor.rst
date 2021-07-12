ndarray()
#########

.. cpp:function:: librapid::basic_ndarray::basic_ndarray()

Create an empty n-dimensional array. This array does not have an
extent or a stride, and many functions will not operate correctly
on such an array. For example, printing out an array created with
this function will result in ``[NONE]`` being printed.

.. Hint::
	No memory is allocated on the heap when using this function,
	so it is incredibly fast
