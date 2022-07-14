#include "librapid/math/mpir.hpp"

namespace librapid {
	/*
	mpf pi() {
		auto bitPrec = mpf_get_default_prec();

		uint64_t q(0);
		mpz n1(545140134);
		mpz n2(13591409);
		mpz n3(-262537412640768000);

		mpf prevFrac(-1);
		mpf frac(0);

		mpz q6fact, n1q, q3fact, qFact, n3q;
		mpz_init2(q6fact.get_mpz_t(), bitPrec);
		mpz_init2(n1q.get_mpz_t(), bitPrec);
		mpz_init2(q3fact.get_mpz_t(), bitPrec);
		mpz_init2(qFact.get_mpz_t(), bitPrec);
		mpz_init2(n3q.get_mpz_t(), bitPrec);

		while (prevFrac != frac) {
			prevFrac = frac;

			// Numerator
			mpz_fac_ui(q6fact.get_mpz_t(), 6 * q);
			n1q = n1 * q;

			// Denominator
			mpz_fac_ui(q3fact.get_mpz_t(), 3 * q);
			mpz_fac_ui(qFact.get_mpz_t(), q);
			qFact *= qFact * qFact; // Cube
			mpz_pow_ui(n3q.get_mpz_t(), n3.get_mpz_t(), q);

			++q;

			frac += mpf(q6fact * (n1q + n2)) / mpf(q3fact * qFact * n3q);
		}

		mpf num(426880);
		num *= sqrt(mpf(10005));

		return num / frac;
	}
	 */

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

	/*
	void bs(const int64_t a, const int64_t b, mpz_class &Pab, mpz_class &Qab, mpz_class &Tab) {
		const mpz_class C_cubed_over_24("10939058860032000"); // C = 640320;

		if (b - a == 1) {
			if (a == 0) {
				Pab = Qab = 1;
			} else {
				Pab = 6 * a - 5;
				Pab *= 2 * a - 1;
				Pab *= 6 * a - 1;
				Qab = C_cubed_over_24;
				Qab *= a;
				Qab *= a;
				Qab *= a;
			}
			Tab = Pab * (13591409 + 545140134 * a);

			if (a & 1) Tab *= -1;

		} else {
			// Binary splitting
			mpz_class Pam, Qam, Tam;
			mpz_class Pmb, Qmb, Tmb;

			int64_t m = (a + b) / 2;

			bs(a, m, Pam, Qam, Tam);
			bs(m, b, Pmb, Qmb, Tmb);

			Pab = Pam * Pmb;
			Qab = Qam * Qmb;
			Tab = Qmb * Tam + Pam * Tmb;
		}
	}

	mpz_class chudnovsky(int64_t digits) {
		digits += 100;

		const double digits_per_term = log10(151931373056000ll); // log(C_cubed_over_24 / 72);
		int64_t N					 = int64_t(digits / digits_per_term) + 1;

		mpz_class P, Q, T;
		bs(0, N, P, Q, T);

		mpz_class one;
		mpz_ui_pow_ui(one.get_mpz_t(), 10, digits);
		mpz_class sqrt_10005 = one * 10005;
		mpz_sqrt(sqrt_10005.get_mpz_t(), sqrt_10005.get_mpz_t());

		mpz_class pi = (Q * 426880 * sqrt_10005) / T;
		return pi;
	}
	 */
} // namespace librapid