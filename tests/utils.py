import librapid

EPSILON = 1e-10

def is_close_numb(a, b):
	return abs(a - b) < EPSILON

def is_close_array(a, b):
	if a.extent != b.extent:
		return False

	return float((a - b).variance()) < 0.01
