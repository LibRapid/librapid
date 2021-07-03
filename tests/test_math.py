import librapid
from . import utils

# Test the math librapidary
def test_math():
	assert librapid.pi == 3.141592653589793238462643383279502884197169399375105820974944592307816406286
	assert librapid.twopi == 6.283185307179586476925286766559005768394338798750211641949889184615632812572
	assert librapid.halfpi == 1.570796326794896619231321691639751442098584699687552910487472296153908203143
	assert librapid.e == 2.718281828459045235360287471352662497757247093699959574966967627724076630353
	assert librapid.sqrt2 == 1.414213562373095048801688724209698078569671875376948073176679737990732478
	assert librapid.sqrt3 == 1.7320508075688772935274463415058723669428052538103806280558069794519330169
	assert librapid.sqrt5 == 2.2360679774997896964091736687312762354406183596115257242708972454105209256378

	assert librapid.product([1, 2, 3, 4, 5]) == 1 * 2 * 3 * 4 * 5
	assert librapid.min([5, 10, 2, 213, -5, 17]) == -5
	assert librapid.max([5, 10, 2, 213, -5, 17]) == 213

	assert librapid.map(5, 0, 10, 0, 1) == 0.5
	assert librapid.map(5, 0, 10, 10, 0) == 5

	for i in range(-5, 6):
		assert librapid.pow10(i) == 10 ** i

	assert utils.is_close_numb(librapid.round(0.5, 0), 1)
	assert utils.is_close_numb(librapid.round(0.44, 1), 0.4)
	assert utils.is_close_numb(librapid.round(0.45, 1), 0.5)
	assert utils.is_close_numb(librapid.round(-0.5, 0), 0)
	assert utils.is_close_numb(librapid.round(-1.4, 0), -1)
	assert utils.is_close_numb(librapid.round(-0.05, 1), 0)
	assert utils.is_close_numb(librapid.round(0.05, 1), 0.1)

	assert utils.is_close_numb(librapid.round(0.123456, 6), 0.123456)
	assert utils.is_close_numb(librapid.round(0.123456, 5), 0.12346)
