import librapid
from . import utils

# Test the ndarray object
def test_ndarray():
	# Test the creation of an array (we have to test 2 things here and assume they work)
	my_array = librapid.ndarray.from_data([1, 2, 3, 4, 5])
	assert str(my_array) == "[1. 2. 3. 4. 5.]"

	assert my_array[0] == 1.
	assert my_array[1] == 2.
	assert my_array[2] == 3.
	assert my_array[3] == 4.
	assert my_array[4] == 5.

	another_array = my_array.clone()
	assert utils.is_close_array(my_array, another_array)

	sum_test_lhs = librapid.ndarray.from_data([[1, 2], [3, 4]])
	sum_test_rhs = librapid.ndarray.from_data([[5, 6], [7, 8]])
	sum_test_result = librapid.ndarray.from_data([[6, 8], [10, 12]])
	sum_test_transposed_result = librapid.ndarray.from_data([[6, 9], [9, 12]])

	assert utils.is_close_array(sum_test_lhs + sum_test_rhs, sum_test_result)
	assert utils.is_close_array(sum_test_lhs.transposed() + sum_test_rhs, sum_test_transposed_result)
