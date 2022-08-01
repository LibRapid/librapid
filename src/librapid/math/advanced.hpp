#pragma once

#include "../internal/config.hpp"
#include "coreMath.hpp"
#include "constants.hpp"

namespace librapid {
	template<typename LAMBDA>
	LR_NODISCARD("")
	LR_INLINE double differentiate(const LAMBDA &fx, double x, double h = 1e-5) {
		double t1 = fx(x - 2 * h) / 12;
		double t2 = 2 * fx(x - h) / 3;
		double t3 = 2 * fx(x + h) / 3;
		double t4 = fx(x + 2 * h) / 12;
		return (1 / h) * (t1 - t2 + t3 - t4);
	}

	template<typename LAMBDA>
	LR_NODISCARD("")
	LR_INLINE double integrate(const LAMBDA &fx, double lower, double upper, double inc = 1e-6) {
		double sum	= inc * inc; // Small error correction
		auto blocks = (int64_t)((upper - lower) / inc);
		for (int64_t i = 0; i < blocks; ++i) {
			double tmp = fx(inc * (double)i) * inc;
			if (std::isinf(tmp)) {
				sum += inc; // Big number?
			} else {
				sum += tmp;
			}
		}
		return sum;
	}

	namespace gammaImpl {
		static int64_t elemsP					  = 8;
		static LR_INLINE std::complex<double> p[] = {676.5203681218851,
													 -1259.1392167224028,
													 771.32342877765313,
													 -176.61502916214059,
													 12.507343278686905,
													 -0.13857109526572012,
													 9.9843695780195716e-6,
													 1.5056327351493116e-7};

		static double epsilon = 1e-7;
		LR_NODISCARD("") LR_INLINE auto dropImag(const std::complex<double> &z) {
			if (abs(z.imag()) < epsilon) std::complex<double>(z.real());
			return z;
		}

		template<typename T>
		LR_NODISCARD("")
		LR_INLINE double gamma(T z_) {
			auto z = std::complex<double>(z_);
			std::complex<double> y;
			if (z.real() < 0.5) {
				y = PI / (sin(PI * z) * gamma(std::complex<double>(1) - z));
			} else {
				z -= 1;
				std::complex<double> x = 0.99999999999980993;
				for (int64_t i = 0; i < elemsP; ++i) {
					auto pVal = p[i];
					x += std::complex<double>(pVal) /
						 (z + std::complex<double>(i) + std::complex<double>(1));
				}
				auto t = z + std::complex<double>((double)elemsP) - std::complex<double>(0.5);
				y	   = sqrt(2 * PI) * pow(t, z + 0.5) * exp(-t) * x;
			}

			return dropImag(y).real();
		}
	} // namespace gammaImpl

	LR_NODISCARD("") LR_INLINE double gamma(double x) {
		LR_ASSERT(x < 143, "Gamma(x = {}) exceeds 64bit floating point range when x >= 143", x);
		return gammaImpl::gamma(x);
	}

	LR_NODISCARD("") LR_INLINE double digamma(double z) {
		double sum = 0;
		for (int64_t k = 0; k < 7500; ++k) { sum += (z - 1) / ((double)(k + 1) * ((double)k + z)); }
		return -EULERMASCHERONI + sum;
	}

	LR_NODISCARD("") LR_INLINE double polygamma(int64_t n, double z, int64_t lim = 100) {
		if (n == 0) return digamma(z);

		double t1	= n & 1 ? 1 : -1;
		double fact = gamma(n - 1);
		double sum	= 0;
		for (int64_t k = 0; k < lim; ++k) { sum += 1 / pow(z + k, n + 1); }
		return t1 * fact * sum;
	}

	LR_NODISCARD("") LR_INLINE double lambertW(double z) {
		/*
		 * Lambert W function, principal branch.
		 * See http://en.wikipedia.org/wiki/Lambert_W_function
		 * Code taken from http://keithbriggs.info/software.html
		 */

		double eps = 4.0e-16;
		double em1 = 0.3678794411714423215955237701614608;
		LR_ASSERT(z >= -em1, "Invalid argument to Lambert W function");

		if (z == 0) return 0;

		if (z < -em1 + 1e-4) {
			double q  = z + em1;
			double r  = sqrt(q);
			double q2 = q * q;
			double q3 = q2 * q;

			// clang-format off
			return -1.0 +
				   2.331643981597124203363536062168 * r -
				   1.812187885639363490240191647568 * q +
				   1.936631114492359755363277457668 * r * q -
				   2.353551201881614516821543561516 * q2 +
				   3.066858901050631912893148922704 * r * q2 -
				   4.175335600258177138854984177460 * q3 +
				   5.858023729874774148815053846119 * r * q3 -
				   8.401032217523977370984161688514 * q3 * q;
			// clang-format on
		}

		double p, w;

		if (z < 1) {
			p = sqrt(2.0 * (2.7182818284590452353602874713526625 * z + 1.0));
			w = -1.0 + p * (1.0 + p * (-0.333333333333333333333 + p * 0.152777777777777777777777));
		} else {
			w = log(z);
		}

		if (z > 3) w -= log(w);

		for (int64_t i = 0; i < 10; ++i) {
			double e = exp(w);
			double t = w * e - z;
			p		 = w + 1;
			t /= e * p - 0.5 * (p + 1.0) * t / p;
			w -= t;
			if (abs(t) < eps * (1 + abs(w))) return w;
		}

		LR_ASSERT(z >= -em1, "Invalid argument to Lambert W function");
		return 0;
	}

	double LR_INLINE invGamma(double x, int64_t prec = 5) {
		// Run a very coarse calculation to get a guess for the guess
		double guess = 2;
		// double tmp	 = gamma(guess);
		// while (abs(gamma(guess) / x) > 0.5) guess += (x < tmp) ? 1 : -1;

		double dx = DBL_MAX;
		while (abs(dx) > pow10(-prec - 1)) {
			double gammaGuess = gamma(guess);
			double num		  = gammaGuess - x;
			double den		  = gammaGuess * polygamma(0, guess);
			double frac		  = num / den;
			double newGuess	  = guess - frac;
			dx				  = guess - newGuess;

			// Avoid nan problems
			if (newGuess > 142) {
				if (newGuess > guess)
					guess *= 2;
				else
					guess /= 2;

				if (guess > 142) guess = 140;
			} else {
				guess = newGuess;
			}
		}
		return round(guess, prec);
	}
} // namespace librapid
