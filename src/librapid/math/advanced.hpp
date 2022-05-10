#pragma once

#include "../internal/config.hpp"
#include "coreMath.hpp"
#include "constants.hpp"

namespace librapid {
	template<typename LAMBDA>
	LR_NODISCARD("")
	double differentiate(const LAMBDA &fx, double x, double h = 1e-5) {
		double t1 = fx(x - 2 * h) / 12;
		double t2 = 2 * fx(x - h) / 3;
		double t3 = 2 * fx(x + h) / 3;
		double t4 = fx(x + 2 * h) / 12;
		return (1 / h) * (t1 - t2 + t3 - t4);
	}

	template<typename LAMBDA>
	LR_NODISCARD("")
	double integrate(const LAMBDA &fx, double lower, double upper, double inc = 1e-6) {
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

	LR_NODISCARD("") double gamma(double z, double upper = -1, double inc = 0.0001) {
		if (upper == -1) {
			if (z < 10) upper = 50;
			if (z < 100) upper = 650;
			if (z >= 100) upper = 1000;
		}

		auto integrand = [&](double x) { return pow(x, z - 1) * exp(-x); };
		return integrate(integrand, 0, upper, inc);
	}

	LR_NODISCARD("") double digamma(double z) {
		double sum = 0;
		for (int64_t k = 0; k < 7500; ++k) { sum += (z - 1) / ((double)(k + 1) * ((double)k + z)); }
		return -EULERMASCHERONI + sum;
	}

	LR_NODISCARD("") double polygamma(int64_t n, double z, int64_t lim = 100) {
		if (n == 0) return digamma(z);

		double t1	= n & 1 ? 1 : -1;
		double fact = gamma(n - 1);
		double sum	= 0;
		for (int64_t k = 0; k < lim; ++k) { sum += 1 / pow<double>(z + k, n + 1); }
		return t1 * fact * sum;
	}

	LR_NODISCARD("") double lambertW(double z) {
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
			w = ln(z);
		}

		if (z > 3) w -= ln(w);

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

	double invGamma(double x, double guess = 2) {
		// Run a very coarse calculation to get a guess for the guess
		while (gamma(guess + 1) < x) guess += 0.5;

		double dx = 10000;
		while (abs(dx) > 1e-10) {
			double num		= gamma(guess + 1) - x;
			double den		= gamma(guess + 1) * polygamma(0, guess + 1);
			double frac		= num / den;
			double newGuess = guess - frac;
			dx				= guess - newGuess;
			guess			= newGuess;
		}
		return guess;
	}
} // namespace librapid
