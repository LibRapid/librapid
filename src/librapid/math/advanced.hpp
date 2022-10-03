#pragma once

namespace librapid {
	template<typename LAMBDA>
	LR_NODISCARD("")
	LR_INLINE f64 differentiate(const LAMBDA &fx, f64 x, f64 h = 1e-5) {
		f64 t1 = fx(x - 2 * h) / 12;
		f64 t2 = 2 * fx(x - h) / 3;
		f64 t3 = 2 * fx(x + h) / 3;
		f64 t4 = fx(x + 2 * h) / 12;
		return (1 / h) * (t1 - t2 + t3 - t4);
	}

	template<typename LAMBDA>
	LR_NODISCARD("")
	LR_INLINE f64 integrate(const LAMBDA &fx, f64 lower, f64 upper, f64 inc = 1e-6) {
		f64 sum		= inc * inc; // Small error correction
		auto blocks = (i64)((upper - lower) / inc);
		for (i64 i = 0; i < blocks; ++i) {
			f64 tmp = fx(inc * (f64)i) * inc;
			if (std::isinf(tmp)) {
				sum += inc; // Big number?
			} else {
				sum += tmp;
			}
		}
		return sum;
	}

	namespace gammaImpl {
		static i64 elemsP					   = 8;
		static LR_INLINE std::complex<f64> p[] = {676.5203681218851,
												  -1259.1392167224028,
												  771.32342877765313,
												  -176.61502916214059,
												  12.507343278686905,
												  -0.13857109526572012,
												  9.9843695780195716e-6,
												  1.5056327351493116e-7};

		static f64 epsilon = 1e-7;
		LR_NODISCARD("") LR_INLINE auto dropImag(const std::complex<f64> &z) {
			if (abs(z.imag()) < epsilon) std::complex<f64>(z.real());
			return z;
		}

		template<typename T>
		LR_NODISCARD("")
		LR_INLINE f64 gamma(T z_) {
			auto z = std::complex<f64>(z_);
			std::complex<f64> y;
			if (z.real() < 0.5) {
				y = PI / (sin(PI * z) * gamma(std::complex<f64>(1) - z));
			} else {
				z -= 1;
				std::complex<f64> x = 0.99999999999980993;
				for (i64 i = 0; i < elemsP; ++i) {
					auto pVal = p[i];
					x +=
					  std::complex<f64>(pVal) / (z + std::complex<f64>(i) + std::complex<f64>(1));
				}
				auto t = z + std::complex<f64>((f64)elemsP) - std::complex<f64>(0.5);
				y	   = sqrt(2 * PI) * pow(t, z + 0.5) * exp(-t) * x;
			}

			return dropImag(y).real();
		}
	} // namespace gammaImpl

	LR_NODISCARD("") LR_INLINE f64 gamma(f64 x) {
		LR_ASSERT(x < 143, "Gamma(x = {}) exceeds 64bit floating point range when x >= 143", x);
		return gammaImpl::gamma(x);
	}

	LR_NODISCARD("") LR_INLINE f64 digamma(f64 z) {
		f64 sum = 0;
		for (i64 k = 0; k < 7500; ++k) { sum += (z - 1) / ((f64)(k + 1) * ((f64)k + z)); }
		return -EULERMASCHERONI + sum;
	}

	LR_NODISCARD("") LR_INLINE f64 polygamma(i64 n, f64 z, i64 lim = 100) {
		if (n == 0) return digamma(z);

		f64 t1	 = n & 1 ? 1 : -1;
		f64 fact = gamma((f64)n - 1);
		f64 sum	 = 0;
		for (i64 k = 0; k < lim; ++k) { sum += 1 / ::librapid::pow(z + (f64)k, n + 1); }
		return t1 * fact * sum;
	}

	LR_NODISCARD("") LR_INLINE f64 lambertW(f64 z) {
		/*
		 * Lambert W function, principal branch.
		 * See http://en.wikipedia.org/wiki/Lambert_W_function
		 * Code taken from http://keithbriggs.info/software.html
		 */

		f64 eps = 4.0e-16;
		f64 em1 = 0.3678794411714423215955237701614608;
		LR_ASSERT(z >= -em1, "Invalid argument to Lambert W function");

		if (z == 0) return 0;

		if (z < -em1 + 1e-4) {
			f64 q  = z + em1;
			f64 r  = sqrt(q);
			f64 q2 = q * q;
			f64 q3 = q2 * q;

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

		f64 p, w;

		if (z < 1) {
			p = sqrt(2.0 * (2.7182818284590452353602874713526625 * z + 1.0));
			w = -1.0 + p * (1.0 + p * (-0.333333333333333333333 + p * 0.152777777777777777777777));
		} else {
			w = log(z);
		}

		if (z > 3) w -= log(w);

		for (i64 i = 0; i < 10; ++i) {
			f64 e = exp(w);
			f64 t = w * e - z;
			p	  = w + 1;
			t /= e * p - 0.5 * (p + 1.0) * t / p;
			w -= t;
			if (abs(t) < eps * (1 + abs(w))) return w;
		}

		LR_ASSERT(z >= -em1, "Invalid argument to Lambert W function");
		return 0;
	}

	f64 LR_INLINE invGamma(f64 x, i64 prec = 5) {
		// Run a very coarse calculation to get a guess for the guess
		f64 guess = 2;
		// f64 tmp	 = gamma(guess);
		// while (abs(gamma(guess) / x) > 0.5) guess += (x < tmp) ? 1 : -1;

		f64 dx = DBL_MAX;
		while (abs(dx) > pow10(-prec - 1)) {
			f64 gammaGuess = gamma(guess);
			f64 num		   = gammaGuess - x;
			f64 den		   = gammaGuess * polygamma(0, guess);
			f64 frac	   = num / den;
			f64 newGuess   = guess - frac;
			dx			   = guess - newGuess;

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
