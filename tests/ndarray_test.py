import librapid as libr

EPSILON = 1e-10

def is_close_numb(a, b):
	return abs(a - b) < EPSILON

# Test the math library
def test_math():
	assert libr.pi == 3.141592653589793238462643383279502884197169399375105820974944592307816406286
	assert libr.twopi == 6.283185307179586476925286766559005768394338798750211641949889184615632812572
	assert libr.halfpi == 1.570796326794896619231321691639751442098584699687552910487472296153908203143
	assert libr.e == 2.718281828459045235360287471352662497757247093699959574966967627724076630353
	assert libr.sqrt2 == 1.414213562373095048801688724209698078569671875376948073176679737990732478
	assert libr.sqrt3 == 1.7320508075688772935274463415058723669428052538103806280558069794519330169
	assert libr.sqrt5 == 2.2360679774997896964091736687312762354406183596115257242708972454105209256378

	assert libr.product([1, 2, 3, 4, 5]) == 1 * 2 * 3 * 4 * 5
	assert libr.min([5, 10, 2, 213, -5, 17]) == -5
	assert libr.max([5, 10, 2, 213, -5, 17]) == 213

	assert libr.map(5, 0, 10, 0, 1) == 0.5
	assert libr.map(5, 0, 10, 10, 0) == 5

	for i in range(-5, 6):
		assert libr.pow10(i) == 10 ** i

	assert is_close_numb(libr.round(0.5, 0), 1)
	assert is_close_numb(libr.round(0.44, 1), 0.4)
	assert is_close_numb(libr.round(0.45, 1), 0.5)
	assert is_close_numb(libr.round(-0.5, 0), 0)
	assert is_close_numb(libr.round(-1.4, 0), -1)
	assert is_close_numb(libr.round(-0.05, 1), 0)
	assert is_close_numb(libr.round(0.05, 1), 0.1)

	assert is_close_numb(libr.round(0.123456, 6), 0.123456)
	assert is_close_numb(libr.round(0.123456, 5), 0.12346)

# Test the extent object
def test_extent():
	from_list = libr.extent([2, 3, 4])
	from_extent = libr.extent(from_list)
	from_args = libr.extent(2, 3, 4)

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

	to_compress = libr.extent(1, 1, 2, 3, 4, 1, 1)
	compressed = to_compress.compressed()

	assert compressed[0] == 2
	assert compressed[1] == 3
	assert compressed[2] == 4

	assert from_list.ndim == 3

	valid = libr.extent(5)
	not_valid = libr.extent()
	assert valid.is_valid == True
	assert not_valid.is_valid == False

	to_reshape = libr.extent(2, 3, 4)
	to_reshape.reshape([2, 1, 0])

	assert to_reshape[0] == 4
	assert to_reshape[1] == 3
	assert to_reshape[2] == 2

	target = libr.extent(2, 3, 4)
	to_fix = libr.extent(2, -1, 4)
	fixed = to_fix.fix_automatic(2 * 3 * 4)
	assert fixed == target

	assert target.ndim == 3

# Test the stride object
def test_stride():
	from_list = libr.stride([2, 3, 4])
	
	assert from_list == from_list
	assert libr.stride(from_list) == from_list

	assert libr.stride(2, 3, 4) == from_list
	assert libr.stride.from_extent(libr.extent(2, 3)) == libr.stride(3, 1)

	assert from_list[0] == 2
	assert from_list[1] == 3
	assert from_list[2] == 4

	valid = libr.stride(2, 3)
	not_valid = libr.stride()

	assert valid.is_valid == True
	assert not_valid.is_valid == False

	valid.reshape([1, 0])
	assert valid == libr.stride([3, 2])

	trivial = libr.stride.from_extent(libr.extent(2, 3))
	not_trivial = libr.stride([2, 3])

	assert trivial.is_trivial == True
	assert not_trivial.is_trivial == False

# Test the ndarray object
def test_ndarray():
	pass
