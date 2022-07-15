#include <cmath>
#include "librapid/math/mpir.hpp"

namespace librapid {
	Chudnovsky::Chudnovsky(int64_t dig10) {
		DIGITS			= dig10;
		A				= 13591409;
		B				= 545140134;
		C				= 640320;
		D				= 426880;
		E				= 10005;
		DIGITS_PER_TERM = 14.1816474627254776555; // = log(53360^3) / log(10)
		C3_24			= C * C * C / 24;
		N				= (int64_t)((double)DIGITS / DIGITS_PER_TERM);
		PREC			= (int64_t)((double)DIGITS * log2(10));
	}

	detail::PQT Chudnovsky::compPQT(int64_t n1, int64_t n2) const {
		int64_t m;
		detail::PQT res;

		if (n1 + 1 == n2) {
			res.P = (2 * n2 - 1);
			res.P *= (6 * n2 - 1);
			res.P *= (6 * n2 - 5);
			res.Q = C3_24 * n2 * n2 * n2;
			res.T = (A + B * n2) * res.P;
			if ((n2 & 1) == 1) res.T = -res.T;
		} else {
			m				 = (n1 + n2) / 2;
			detail::PQT res1 = compPQT(n1, m);
			detail::PQT res2 = compPQT(m, n2);
			res.P			 = res1.P * res2.P;
			res.Q			 = res1.Q * res2.Q;
			res.T			 = res1.T * res2.Q + res1.P * res2.T;
		}

		return res;
	}

	mpf Chudnovsky::pi() const {
		// Compute Pi
		detail::PQT pqt = compPQT(0, N);
		mpf pi(0, PREC);
		pi = D * sqrt((mpf_class)E) * pqt.Q;
		pi /= (A * pqt.Q + pqt.T);
		return pi;
	}
} // namespace librapid