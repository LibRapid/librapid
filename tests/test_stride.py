import librapid
from . import utils

# Test the stride object
def test_stride():
	from_list = librapid.stride([2, 3, 4])
	
	assert from_list == from_list
	assert librapid.stride(from_list) == from_list

	assert librapid.stride(2, 3, 4) == from_list
	assert librapid.stride.from_extent(librapid.extent(2, 3)) == librapid.stride(3, 1)

	assert from_list[0] == 2
	assert from_list[1] == 3
	assert from_list[2] == 4

	valid = librapid.stride(2, 3)
	not_valid = librapid.stride()

	assert valid.is_valid == True
	assert not_valid.is_valid == False

	valid.reshape([1, 0])
	assert valid == librapid.stride([3, 2])

	trivial = librapid.stride.from_extent(librapid.extent(2, 3))
	not_trivial = librapid.stride([2, 3])

	assert trivial.is_trivial == True
	assert not_trivial.is_trivial == False
