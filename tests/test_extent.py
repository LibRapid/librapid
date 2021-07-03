import librapid
from . import utils

# Test the extent object
def test_extent():
	from_list = librapid.extent([2, 3, 4])
	from_extent = librapid.extent(from_list)
	from_args = librapid.extent(2, 3, 4)

	assert from_list == from_list

	assert from_list[0] == 2
	assert from_list[1] == 3
	assert from_list[2] == 4

	assert from_extent[0] == 2
	assert from_extent[1] == 3
	assert from_extent[2] == 4

	assert from_args[0] == 2
	assert from_args[1] == 3
	assert from_args[2] == 4

	to_compress = librapid.extent(1, 1, 2, 3, 4, 1, 1)
	compressed = to_compress.compressed()

	assert compressed[0] == 2
	assert compressed[1] == 3
	assert compressed[2] == 4

	assert from_list.ndim == 3

	valid = librapid.extent(5)
	not_valid = librapid.extent()
	assert valid.is_valid == True
	assert not_valid.is_valid == False

	to_reshape = librapid.extent(2, 3, 4)
	to_reshape.reshape([2, 1, 0])

	assert to_reshape[0] == 4
	assert to_reshape[1] == 3
	assert to_reshape[2] == 2

	target = librapid.extent(2, 3, 4)
	to_fix = librapid.extent(2, -1, 4)
	fixed = to_fix.fix_automatic(2 * 3 * 4)
	assert fixed == target

	assert target.ndim == 3
