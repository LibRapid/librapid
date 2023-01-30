#include <librapid/librapid.hpp>

namespace librapid::fastmath {
	double pow10(int64_t exponent) {
		const static double pows[] = {0.0000001,
									  0.000001,
									  0.00001,
									  0.0001,
									  0.001,
									  0.01,
									  0.1,
									  1,
									  10,
									  100,
									  1000,
									  10000,
									  100000,
									  1000000,
									  1000000};

		if (exponent >= -7 && exponent <= 7) return pows[exponent + 7];

		double res = 1;

		if (exponent > 0)
			for (int64_t i = 0; i < exponent; ++i) res *= 10.;
		else
			for (int64_t i = 0; i > exponent; --i) res *= 0.1;

		return res;
	}
}