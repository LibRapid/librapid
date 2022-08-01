#pragma once

/*
 * A Complex Number implementation, based loosely off of MSVC's std::complex<T> datatype.
 * This type does not conform to the C++ standard, but it *should* support a wider range of
 * primitive types (*1*) and user-defined types.
 *
 * (*1*) Why on Earth would you create a complex number out of an integer type?
 */

#include "../internal/config.hpp"

#if defined(_M_IX86) || (defined(_M_X64) && !defined(_M_ARM64EC))
#	define USE_X86_X64_INTRINSICS
#	include <emmintrin.h>
#elif defined(_M_ARM64) || defined(_M_ARM64EC)
#	define USE_ARM64_INTRINSICS
#	include <arm64_neon.h>
#endif

namespace librapid {
	namespace detail {
		// The basic complex container type
		template<typename Scalar>
		struct ComplexImpl {
			Scalar val[2];
		};

#if defined(LIBRAPID_USE_VC)
		template<>
		struct ComplexImpl<int16_t> {
			Vc::int16_v val;
		};

		template<>
		struct ComplexImpl<int32_t> {
			Vc::int32_v val;
		};

		template<>
		struct ComplexImpl<int64_t> {
			Vc::int64_t val;
		};

		template<>
		struct ComplexImpl<float> {
			Vc::float_v val;
		};

		template<>
		struct ComplexImpl<double> {
			Vc::double_v val;
		};
#endif

		// Implements floating-point arithmetic for numeric algorithms
		namespace multiprec {
			template<typename Scalar>
			struct Fmp {
				Scalar val0; // Most significant numeric_limits<Scalar>::precision bits
				Scalar val1; // Least significant numeric_limits<Scalar>::precision bits
			};

			// 1x precision + 1x precision -> 2x precision
			// the result is exact when:
			// 1) the result doesn't overflow
			// 2) either underflow is gradual, or no internal underflow occurs
			// 3) intermediate precision is either the same as _Ty, or greater than twice the
			// precision
			//    of _Ty
			// 4) parameters and local variables do not retain extra intermediate precision 5)
			//    rounding mode is rounding to nearest violation of condition 3 or 5 could lead to
			//    relative error on the order of epsilon^2 violation of other conditions could lead
			//    to worse results
			template<typename T>
			LR_NODISCARD("")
			LR_INLINE constexpr Fmp<T> addX2(const T &x, const T &y) noexcept {
				const T sum0 = x + y;
				const T yMod = sum0 - x;
				const T xMod = sum0 - yMod;
				const T yErr = y - yMod;
				const T xErr = x - xMod;
				return {sum0, xErr + yErr};
			}

			// 1x precision + 1x precision -> 2x precision
			// requires: exponent(x) + countr_zero(significand(x)) >= exponent(y) || x
			// == 0 the result is exact when: 0) the requirement above is satisfied 1) no internal
			// overflow occurs 2) either underflow is gradual, or no internal underflow occurs 3)
			// intermediate precision is either the same as _Ty, or greater than twice the precision
			// of _Ty 4) parameters and local variables do not retain extra intermediate precision
			// 5) rounding mode is rounding to nearest violation of condition 3 or 5 could lead to
			// relative error on the order of epsilon^2 violation of other conditions could lead to
			// worse results
			template<typename T>
			LR_NODISCARD("")
			LR_INLINE constexpr Fmp<T> addSmallX2(const T x, const T y) noexcept {
				const T sum0 = x + y;
				const T yMod = sum0 - x;
				const T yErr = y - yMod;
				return {sum0, yErr};
			}

			// 1x precision + 2x precision -> 2x precision
			// requires:
			// exponent(x) + countr_zero(significand(x)) >= exponent(y.val0) || x == 0
			template<typename T>
			LR_NODISCARD("")
			LR_INLINE constexpr Fmp<T> addSmallX2(const T &x, const Fmp<T> &y) noexcept {
				const Fmp<T> sum0 = addSmallX2(x, y.val0);
				return addSmallX2(sum0.val0, sum0.val1 + y.val1);
			}

			// 2x precision + 2x precision -> 1x precision
			template<typename T>
			LR_NODISCARD("")
			LR_INLINE constexpr T addX1(const Fmp<T> &x, const Fmp<T> &y) noexcept {
				const Fmp<T> sum0 = addX2(x.val0, y.val0);
				return sum0.val0 + (sum0.val1 + (x.val1 + y.val1));
			}

			// Round to 26 significant bits. Ties toward zero
			LR_NODISCARD("") LR_INLINE constexpr double highHalf(const double x) noexcept {
				const auto bits			= bitCast<uint64_t>(x);
				const auto highHalfBits = (bits + 0x3ff'ffffULL) & 0xffff'ffff'f800'0000ULL;
				return bitCast<double>(highHalfBits);
			}

#if defined(USE_X86_X64_INTRINSICS) || defined(USE_ARM64_INTRINSICS) // SIMD method
			// x * x - prod0
			// the result is exact when:
			// 1) prod0 is x^2 faithfully rounded
			// 2) no internal overflow or underflow occurs
			// violation of condition 1 could lead to relative error on the order of epsilon
			LR_NODISCARD("")
			LR_INLINE double sqrError(const double x, const double prod0) noexcept {
#	if defined(USE_X86_X64_INTRINSICS)
				const __m128d xVec		= _mm_set_sd(x);
				const __m128d prodVec	= _mm_set_sd(prod0);
				const __m128d resultVec = _mm_fmsub_sd(xVec, xVec, prodVec);
				double result;
				_mm_store_sd(&result, resultVec);
				return result;
#	else // Only two options, so this is fine
				const float64x1_t xVec		= vld1_f64(&x);
				const float64x1_t prod0Vec	= vld1_f64(&prod0);
				const float64x1_t resultVec = vfma_f64(vneg_f64(prod0Vec), xVec, xVec);
				double result;
				vst1_f64(&result, resultVec);
				return result;
#	endif
			}
#else
			LR_NODISCARD("") // Fallback method
			LR_INLINE constexpr double sqrError(const double x, const double prod0) noexcept {
				const double xHigh = highHalf(x);
				const double xLow  = x - xHigh;
				return ((xHigh * xHigh - prod0) + 2.0 * xHigh * xLow) + xLow * xLow;
			}
#endif

			template<typename T>
			LR_NODISCARD("") // Fallback method
			LR_INLINE T sqrError(const T x, const T prod0) noexcept {
				const T xHigh = highHalf(x);
				const T xLow  = x - xHigh;
				return ((xHigh * xHigh - prod0) + 2.0 * xHigh * xLow) + xLow * xLow;
			}

			// square(1x precision) -> 2x precision
			// the result is exact when no internal overflow or underflow occurs
			LR_NODISCARD("") LR_INLINE Fmp<double> sqrX2(const double x) noexcept {
				const double prod0 = x * x;
				return {prod0, sqrError(x, prod0)};
			}

			template<typename T>
			LR_NODISCARD("")
			LR_INLINE Fmp<T> sqrX2(const T x) noexcept {
				const T prod0 = x * x;
				return {prod0, sqrError(x, prod0)};
			}
		} // namespace multiprec

		namespace algorithm {
			// HypotLegHuge = T{0.5} * sqrt((numeric_limits<T>::max()));
			// HypotLegTiny = sqrt(T{2.0} * (numeric_limits<T>::min)() /
			// numeric_limits<T>::epsilon());

			template<typename T>
			struct HypotLegHugeHelper {
				// If <T> is an integer type, divide by two rather than multiplying by 0.5, as
				// 0.5 gets truncated to zero
				static inline T val = (std::is_integral_v<T>)
										? (::librapid::sqrt(internal::traits<T>::max()) / T(2))
										: (T(0.5) * ::librapid::sqrt(internal::traits<T>::max()));
			};

			template<>
			struct HypotLegHugeHelper<double> {
				static constexpr double val = 6.703903964971298e+153;
			};

			template<>
			struct HypotLegHugeHelper<float> {
				static constexpr double val = 9.2233715e+18f;
			};

			template<typename T>
			struct HypotLegTinyHelper {
				// If <T> is an integer type, divide by two rather than multiplying by 0.5, as
				// 0.5 gets truncated to zero
				static inline T val = ::librapid::sqrt(T(2) * internal::traits<T>::min() /
													   internal::traits<T>::epsilon());
			};

			template<>
			struct HypotLegTinyHelper<double> {
				static constexpr double val = 1.4156865331029228e-146;
			};

			template<>
			struct HypotLegTinyHelper<float> {
				static constexpr double val = 4.440892e-16f;
			};

			template<typename T>
			static inline T HypotLegHuge = HypotLegHugeHelper<T>::val;
			template<typename T>
			static inline T HypotLegTiny = HypotLegTinyHelper<T>::val;

			template<typename T>
			LR_NODISCARD("")
			LR_INLINE T normMinusOne(const T x, const T y) noexcept {
				// requires |x| >= |y| and 0.5 <= |x| < 2^12
				// returns x * x + y * y - 1
				const multiprec::Fmp<T> xSqr   = multiprec::sqrX2(x);
				const multiprec::Fmp<T> ySqr   = multiprec::sqrX2(y);
				const multiprec::Fmp<T> xSqrM1 = multiprec::addSmallX2(T(-1), xSqr);
				return multiprec::addX1(xSqrM1, ySqr);
			}

			// Returns log(1 + x)
			// May be inaccurate for small inputs
			template<bool safe = true, typename T>
			LR_NODISCARD("")
			LR_INLINE T logP1(const T x) {
				if constexpr (!safe) return ::librapid::log(x + 1.0);
#if defined(LIBRAPID_USE_MULTIPREC)
				// No point doing anything shown below if we're using multiprec
				if constexpr (std::is_same_v<T, mpfr>) return ::librapid::log(x + 1.0);
#endif

				if (internal::isNaN(x)) return x + x; // Trigger a signaling NaN

				// Naive formula
				if (x <= T(-0.5) || T(2) <= x) {
					// To avoid overflow
					if (x == internal::traits<T>::max()) return ::librapid::log(x);
					return ::librapid::log(T(1) + x);
				}

				const T absX = ::librapid::abs(x);
				if (absX < internal::traits<T>::epsilon()) {
					if (x == T(0)) return x;
					return x - T(0.5) * x * x; // Honour rounding
				}

				// log(1 + x) with fix for small x
				const multiprec::Fmp<T> tmp = multiprec::addSmallX2(T(1), x);
				return ::librapid::log(tmp.val0) + tmp.val1 / tmp.val0;
			}

			// Return log(hypot(x, y))
			template<bool safe = true, typename T>
			LR_NODISCARD("")
			LR_INLINE T logHypot(const T x, const T y) noexcept {
				if constexpr (!safe) return ::librapid::log(::librapid::sqrt(x * x + y * y));
#if defined(LIBRAPID_USE_MULTIPREC)
				// No point doing anything shown below if we're using multiprec
				if constexpr (std::is_same_v<T, mpfr>)
					return ::librapid::log(::mpfr::hypot(x, y));
				else {
#endif

					if (!internal::isFinite(x) || !internal::isFinite(y)) { // Inf or NaN
						// Return NaN and raise FE_INVALID if either x or y is NaN
						if (internal::isNaN(x) || internal::isNaN(y)) return x + y;

						// Return Inf if either of them is infinity
						if (internal::isInf(x)) return x;
						if (internal::isInf(y)) return y;

						return x + y; // Fallback
					}

					T absX = ::librapid::abs(x);
					T absY = ::librapid::abs(y);

					if (absX < absY) std::swap(absX, absY);		 // Ensure absX > absY
					if (absY == 0) return ::librapid::log(absX); // One side has zero length

					// Avoid overflow and underflow
					if (HypotLegTiny<T> < absX && absX < HypotLegHuge<T>) {
						constexpr auto normSmall = T(0.5);
						constexpr auto normBig	 = T(3.0);

						const T absYSqr = absY * absY;

						if (absX == T(1)) return logP1(absYSqr) * T(0.5);

						const T norm = absX * absX + absYSqr;
						if (normSmall < norm && norm < normBig) // Avoid cancellation
							return logP1(normMinusOne(absX, absY)) * T(0.5);
						return ::librapid::log(norm) * T(0.5);
					} else { // Use 1 1/2 precision to preserve bits
						constexpr T cm = T(22713.0L / 32768.0L); // Not sure where this came from
						constexpr T cl = T(1.4286068203094172321214581765680755e-6L); // Or this...

						const int exp		  = std::ilogb(absX);
						const T absXScaled	  = std::scalbn(absX, -exp);
						const T absYScaled	  = std::scalbn(absY, -exp);
						const T absYScaledSqr = absYScaled * absYScaled;
						const T normScaled	  = absXScaled * absXScaled + absYScaledSqr;
						const T realShifted	  = ::librapid::log(normScaled) * T(0.5);
						const auto fExp		  = static_cast<T>(exp);
						return (realShifted + fExp * cl) + fExp * cm;
					}
#if defined(LIBRAPID_USE_MULTIPREC)
				} // This ensures the "if constexpr" above actually stops compiler errors
#endif
			}
		} // namespace algorithm
	}	  // namespace detail

	template<typename T>
	class Complex {
	public:
		constexpr Complex(const T &realVal = 0, const T &imagVal = 0) : m_val {realVal, imagVal} {}

		template<typename Other>
		constexpr Complex(const Complex<Other> &other) : m_val {other.real(), other.imag()} {}

		LR_INLINE void real(const T &val) { m_val[RE] = val; }
		LR_INLINE void imag(const T &val) { m_val[IM] = val; }
		LR_INLINE constexpr T real() const { return m_val[RE]; }
		LR_INLINE constexpr T imag() const { return m_val[IM]; }

		LR_INLINE Complex &operator=(const T &other) {
			m_val[RE] = other;
			m_val[IM] = 0;
			return *this;
		}

		template<typename Other>
		LR_INLINE Complex &operator=(const Complex<Other> &other) {
			m_val[RE] = internal::traits<Other>::template cast<T>(other.real());
			m_val[IM] = internal::traits<Other>::template cast<T>(other.real());
			return *this;
		}

		LR_INLINE Complex &operator+=(const T &other) {
			m_val[RE] = m_val[RE] + other;
			return *this;
		}

		LR_INLINE Complex &operator-=(const T &other) {
			m_val[RE] = m_val[RE] - other;
			return *this;
		}

		LR_INLINE Complex &operator*=(const T &other) {
			m_val[RE] = m_val[RE] * other;
			m_val[IM] = m_val[IM] * other;
			return *this;
		}

		LR_INLINE Complex &operator/=(const T &other) {
			m_val[RE] = m_val[RE] / other;
			m_val[IM] = m_val[IM] / other;
			return *this;
		}

		LR_INLINE Complex &operator+=(const Complex &other) {
			this->_add(other);
			return *this;
		}

		LR_INLINE Complex &operator-=(const Complex &other) {
			this->_sub(other);
			return *this;
		}

		LR_INLINE Complex &operator*=(const Complex &other) {
			this->_mul(other);
			return *this;
		}

		LR_INLINE Complex &operator/=(const Complex &other) {
			this->_div(other);
			return *this;
		}

	protected:
		template<typename Other>
		LR_INLINE void _add(const Complex<Other> &other) {
			m_val[RE] = m_val[RE] + other.real();
			m_val[IM] = m_val[IM] + other.imag();
		}

		template<typename Other>
		LR_INLINE void _sub(const Complex<Other> &other) {
			m_val[RE] = m_val[RE] - other.real();
			m_val[IM] = m_val[IM] - other.imag();
		}

		template<typename Other>
		LR_INLINE void _mul(const Complex<Other> &other) {
			T otherReal = internal::traits<Other>::template cast<T>(other.real());
			T otherImag = internal::traits<Other>::template cast<T>(other.imag());

			T tmp	  = m_val[RE] * otherReal - m_val[IM] * otherImag;
			m_val[IM] = m_val[RE] * otherImag + m_val[IM] * otherReal;
			m_val[RE] = tmp;
		}

		template<typename Other>
		LR_INLINE void _div(const Complex<Other> &other) {
			T otherReal = internal::traits<Other>::template cast<T>(other.real());
			T otherImag = internal::traits<Other>::template cast<T>(other.imag());

			if (internal::isNaN(otherReal) || internal::isNaN(otherImag)) { // Set result to NaN
				m_val[RE] = internal::traits<T>::quietNaN();
				m_val[IM] = m_val[RE];
			} else if ((otherImag < 0 ? -otherImag
									  : +otherImag) < // |other.imag()| < |other.real()|
					   (otherReal < 0 ? -otherReal : +otherReal)) {
				T wr = otherImag / otherReal;
				T wd = otherReal + wr * otherImag;

				if (internal::isNaN(wd) || wd == 0) { // NaN result
					m_val[RE] = internal::traits<T>::quietNaN();
					m_val[IM] = m_val[RE];
				} else { // Valid result
					T tmp	  = (m_val[RE] + m_val[IM] * wr) / wd;
					m_val[IM] = (m_val[IM] - m_val[RE] * wr) / wd;
					m_val[RE] = tmp;
				}
			} else if (otherImag == 0) { // Set NaN
				m_val[RE] = internal::traits<T>::quietNaN();
				m_val[IM] = m_val[RE];
			} else { // 0 < |other.real()| <= |other.imag()|
				T wr = otherReal / otherImag;
				T wd = otherImag + wr * otherReal;

				if (internal::isNaN(wd) || wd == 0) { // NaN result
					m_val[RE] = internal::traits<T>::quietNaN();
					m_val[IM] = m_val[RE];
				} else {
					T tmp	  = (m_val[RE] * wr + m_val[IM]) / wd;
					m_val[IM] = (m_val[IM] * wr - m_val[RE]) / wd;
					m_val[RE] = tmp;
				}
			}
		}

	private:
		T m_val[2];
		static constexpr size_t RE = 0;
		static constexpr size_t IM = 1;
	};

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator+(const Complex<T> &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp += right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator+(const Complex<T> &left, const T &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() + right);
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator+(const T &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp += right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator-(const Complex<T> &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp -= right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator-(const Complex<T> &left, const T &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() - right);
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator-(const T &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp -= right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator*(const Complex<T> &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp *= right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator*(const Complex<T> &left, const T &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() * right);
		tmp.imag(tmp.imag() * right);
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator*(const T &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp *= right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator/(const Complex<T> &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp /= right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator/(const Complex<T> &left, const T &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() / right);
		tmp.imag(tmp.imag() / right);
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> operator/(const T &left, const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp /= right;
		return tmp;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE constexpr bool operator==(const Complex<T> &left, const Complex<T> &right) {
		return left.real() == right.real() && left.imag() == right.imag();
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE constexpr bool operator==(const Complex<T> &left, T &right) {
		return left.real() == right && left.imag() == 0;
	}

#if !defined(LIBRAPID_CXX_20)
	template<typename T>
	LR_NODISCARD("")
	LR_INLINE constexpr bool operator==(const T &left, const Complex<T> &right) {
		return left == right.real() && 0 == right.imag();
	}
#endif

#if !defined(LIBRAPID_CXX_20)
	template<typename T>
	LR_NODISCARD("")
	LR_INLINE constexpr bool operator!=(const Complex<T> &left, const Complex<T> &right) {
		return !(left == right);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE constexpr bool operator!=(const Complex<T> &left, T &right) {
		return !(left == right);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE constexpr bool operator!=(const T &left, const Complex<T> &right) {
		return !(left == right);
	}
#endif

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> sqrt(const Complex<T> &val); // Defined later

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> abs(const Complex<T> &val) {
		return ::librapid::hypot(val.real(), val.imag());
	}
} // namespace librapid

#if defined(FMT_API)
template<typename T>
struct fmt::formatter<librapid::Complex<T>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	inline auto format(const librapid::Complex<T> &num, FormatContext &ctx) {
		try {
			std::string real = fmt::format(formatStr, num.real());
			std::string imag = fmt::format(formatStr, (num.imag() < 0 ? -num.imag() : num.imag()));
			std::string ret;
			if (num.imag() >= 0)
				ret = fmt::format("({}+{}j)", real, imag);
			else
				ret = fmt::format("({}-{}j)", real, imag);

			return fmt::format_to(ctx.out(), ret);
		} catch (std::exception &e) {
			LR_ASSERT("Invalid Format Specifier: {}", e.what());
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};
#endif // FMT_API

#undef USE_X86_X64_INTRINSICS
#undef USE_ARM64_INTRINSICS