#ifndef LIBRAPID_ARRAY_MATH_UTILS
#define LIBRAPID_ARRAY_MATH_UTILS

#include <librapid/math/rapid_math.hpp>
#include <cmath>

namespace librapid {
	class Array;

	Array sin(const Array &arr);
	Array cos(const Array &arr);
	Array tan(const Array &arr);

	Array asin(const Array &arr);
	Array acos(const Array &arr);
	Array atan(const Array &arr);

	Array sinh(const Array &arr);
	Array cosh(const Array &arr);
	Array tanh(const Array &arr);

	Array asinh(const Array &arr);
	Array acosh(const Array &arr);
	Array atanh(const Array &arr);

	Array pow(const Array &arr, double exponent);
	Array sqrt(const Array &arr);
	Array exp(const Array &arr);
}

#endif // LIBRAPID_ARRAY_MATH_UTILS