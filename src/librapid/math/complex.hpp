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

			// Compute exp(*pleft) * right * 2 ^ exponent
			template<typename T>
			short expMul(T *pleft, T right, short exponent) {
#if defined(LIBRAPID_USE_MULTIPREC)
				if constexpr (std::is_same_v<T, mpfr>) {
					*pleft = ::mpfr::exp(*pleft) * right * ::mpfr::exp2(exponent);
					return (internal::isNaN(*pleft) || internal::isInf(*pleft)) ? 1 : -1;
				} else {
#endif

#if defined(LIBRAPID_MSVC_CXX)
					auto tmp  = static_cast<double>(*pleft);
					short ans = _CSTD _Exp(&tmp, static_cast<double>(right), exponent);
					*pleft	  = static_cast<T>(tmp);
					return ans;
#else
				*pleft = ::librapid::exp(*pleft) * right * ::librapid::exp2(exponent);
				return (internal::isNaN(*pleft) || internal::isInf(*pleft)) ? 1 : -1;
#endif

#if defined(LIBRAPID_USE_MULTIPREC)
				} // This ensures the "if constexpr" above actually stops compiler errors
#endif
			}
		} // namespace algorithm
	}	  // namespace detail

	template<typename T = double>
	class Complex {
	public:
		template<typename S = T>
		Complex(const S &realVal = 0, const S &imagVal = 0) : m_val {T(realVal), T(imagVal)} {}
		Complex(const Complex<T> &other) : m_val {other.real(), other.imag()} {}
		Complex(Complex<T> &&other) : m_val {other.real(), other.imag()} {}
		Complex(const std::complex<T> &other) : m_val {other.real(), other.imag()} {}

		Complex<T> &operator=(const Complex<T> &other) {
			if (this == &other) return *this;
			m_val[RE] = other.real();
			m_val[IM] = other.imag();
			return *this;
		}

		LR_INLINE void real(const T &val) { m_val[RE] = val; }
		LR_INLINE void imag(const T &val) { m_val[IM] = val; }
		LR_INLINE const T &real() const { return m_val[RE]; }
		LR_INLINE const T &imag() const { return m_val[IM]; }
		LR_INLINE T &real() { return m_val[RE]; }
		LR_INLINE T &imag() { return m_val[IM]; }

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

		template<typename To>
		LR_INLINE operator To() const {
			return internal::traits<T>::template cast<To>(m_val[RE]);
		}

		template<typename To>
		LR_INLINE operator Complex<To>() const {
			return Complex<To>(internal::traits<T>::template cast<To>(m_val[RE]),
							   internal::traits<T>::template cast<To>(m_val[IM]));
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
	LR_INLINE Complex<T> operator-(const Complex<T> &other) {
		return Complex<T>(-other.real(), -other.imag());
	}

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
	LR_INLINE T real(const Complex<T> &val) {
		return val.real();
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T imag(const Complex<T> &val) {
		return val.imag();
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> sqrt(const Complex<T> &val); // Defined later

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T abs(const Complex<T> &val) {
		return ::librapid::hypot(val.real(), val.imag());
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> conj(const Complex<T> &val) {
		return Complex<T>(val.real(), -val.imag());
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> acos(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(internal::traits<T>::max());
		const T pi	   = []() {
#if defined(LIBRAPID_USE_MULTIPREC)
			if constexpr (std::is_same_v<T, mpfr>)
				return ::librapid::constPi();
			else
				return static_cast<T>(3.1415926535897932384626433832795029L);
#else
			return static_cast<T>(3.1415926535897932384626433832795029L);
#endif
		}();

		const T re = real(other);
		const T im = imag(other);
		T ux, vx;

		if (internal::isNaN(re) || internal::isNaN(im)) { // At least one NaN
			ux = internal::traits<T>::quietNaN();
			vx = ux;
		} else if (internal::isInf(re)) { // +/- Inf
			if (internal::isInf(im)) {
				if (re < 0)
					ux = T(0.75) * pi; // (-Inf, +/-Inf)
				else
					ux = T(0.25) * pi; // (-Inf, +/-Inf)
			} else if (re < 0) {
				ux = pi; // (-Inf, finite)
			} else {
				ux = 0; // (+Inf, finite)
			}
			vx = -internal::copySign(internal::traits<T>::infinity(), im);
		} else if (internal::isInf(im)) { // finite, Inf)
			ux = T(0.5) * pi;			  // (finite, +/-Inf)
			vx = -im;
		} else { // (finite, finite)
			const Complex<T> wx = sqrt(Complex<T>(1 + re, -im));
			const Complex<T> zx = sqrt(Complex<T>(1 - re, -im));
			const T wr			= real(wx);
			const T wi			= imag(wx);
			const T zr			= real(zx);
			const T zi			= imag(zx);
			T alpha, beta;

			ux = 2 * ::librapid::atan2(zr, wr);

			if (arcBig < wr) { // Real part is large
				alpha = wr;
				beta  = zi + wi * (zr / alpha);
			} else if (arcBig < wi) { // Imaginary part is large
				alpha = wi;
				beta  = wr * (zi / alpha) + zr;
			} else if (wi < -arcBig) { // Imaginary part of w is large negative
				alpha = -wi;
				beta  = wr * (zi / alpha) - zr;
			} else { // Shouldn't overflow (?)
				alpha = 0;
				beta  = wr * zi + wi * zr; // Im(w * z)
			}

			vx = ::librapid::asinh(beta);
			if (alpha != 0) {
				// asinh(a * b) = asinh(a) + log(b)
				if (0 <= vx)
					vx += ::librapid::log(alpha);
				else
					vx -= ::librapid::log(alpha);
			}
		}
		return Complex<T>(ux, vx);
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> acosh(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(internal::traits<T>::max());
		const T pi	   = []() {
#if defined(LIBRAPID_USE_MULTIPREC)
			if constexpr (std::is_same_v<T, mpfr>)
				return ::librapid::constPi();
			else
				return static_cast<T>(3.1415926535897932384626433832795029L);
#else
			return static_cast<T>(3.1415926535897932384626433832795029L);
#endif
		}();

		const T re = real(other);
		T im	   = imag(other);
		T ux, vx;

		if (internal::isNaN(re) || internal::isNaN(im)) { // At least one NaN
			ux = internal::traits<T>::quietNaN();
			vx = ux;
		} else if (internal::isInf(re)) { // (+/-Inf, not NaN)
			ux = internal::traits<T>::infinity();
			if (internal::isInf(im)) {
				if (re < 0)
					vx = T(0.75) * pi; // (-Inf, +/-Inf)
				else
					vx = T(0.25) * pi; // (+Inf, +/-Inf)
			} else if (re < 0) {
				vx = pi; // (-Inf, finite)
			} else {
				vx = 0; // (+Inf, finite)
			}
			vx = internal::copySign(vx, im);
		} else { // (finite, finite)
			const Complex<T> wx = sqrt(Complex<T>(re - 1, -im));
			const Complex<T> zx = sqrt(Complex<T>(re + 1, im));
			const T wr			= real(wx);
			const T wi			= imag(wx);
			const T zr			= real(zx);
			const T zi			= imag(zx);
			T alpha, beta;

			if (arcBig < wr) { // Real parts large
				alpha = wr;
				beta  = zr - wi * (zi / alpha);
			} else if (arcBig < wi) { // Imaginary parts large
				alpha = wi;
				beta  = wr * (zr / alpha) - zi;
			} else { // Shouldn't overflow (?)
				alpha = 0;
				beta  = wr * zr - wi * zi; // Re(w * z)
			}

			ux = ::librapid::asinh(beta);
			if (alpha != 0) {
				if (0 <= ux)
					ux += ::librapid::log(alpha);
				else
					ux -= ::librapid::log(alpha);
			}
			vx = 2 * ::librapid::atan2(imag(sqrt(Complex<T>(re - 1, im))), zr);
		}
		return Complex<T>(ux, vx);
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> asinh(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(internal::traits<T>::max());
		const T pi	   = []() {
#if defined(LIBRAPID_USE_MULTIPREC)
			if constexpr (std::is_same_v<T, mpfr>)
				return ::librapid::constPi();
			else
				return static_cast<T>(3.1415926535897932384626433832795029L);
#else
			return static_cast<T>(3.1415926535897932384626433832795029L);
#endif
		}();

		const T re = real(other);
		T im	   = imag(other);
		T ux, vx;

		if (internal::isNaN(re) || internal::isNaN(im)) { // At least one NaN/Inf
			ux = internal::traits<T>::quietNaN();
			vx = ux;
		} else if (internal::isInf(re)) { // (+/-Inf, not NaN)
			if (internal::isInf(im)) {	  // (+/-Inf, +/-Inf)
				ux = re;
				vx = internal::copySign(T(0.25) * pi, im);
			} else { // (+/-Inf, finite)
				ux = re;
				vx = internal::copySign(T(0), im);
			}
		} else if (internal::isInf(im)) {
			ux = internal::copySign(internal::traits<T>::infinity(), re);
			vx = internal::copySign(T(0.5) * pi, im);
		} else { // (finite, finite)
			const Complex<T> wx = sqrt(Complex<T>(1 - im, re));
			const Complex<T> zx = sqrt(Complex<T>(1 + im, -re));
			const T wr			= real(wx);
			const T wi			= imag(wx);
			const T zr			= real(zx);
			const T zi			= imag(zx);
			T alpha, beta;

			if (arcBig < wr) { // Real parts are large
				alpha = wr;
				beta  = wi * (zr / alpha) - zi;
			} else if (arcBig < wi) { // Imaginary parts are large
				alpha = wi;
				beta  = zr - wr * (zi / alpha);
			} else if (wi < -arcBig) {
				alpha = -wi;
				beta  = -zr - wr * (zi / alpha);
			} else { // Shouldn't overflow (?)
				alpha = 0;
				beta  = wi * zr - wr * zi; // Im(w * conj(z))
			}

			ux = ::librapid::asinh(beta);
			if (alpha != 0) {
				if (0 <= ux)
					ux += ::librapid::log(alpha);
				else
					ux -= ::librapid::log(alpha);
			}
			vx = ::librapid::atan2(im, real(wx * zx));
		}
		return Complex<T>(ux, vx);
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> asin(const Complex<T> &other) {
		Complex<T> asinhVal = asinh(Complex<T>(-imag(other), real(other)));
		return Complex<T>(imag(asinhVal), -real(asinhVal));
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> atanh(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(internal::traits<T>::max());
		const T piBy2  = []() {
#if defined(LIBRAPID_USE_MULTIPREC)
			if constexpr (std::is_same_v<T, mpfr>)
				return ::librapid::constPi() / 2;
			else
				return static_cast<T>(1.5707963267948966192313216916397514L);
#else
			return static_cast<T>(1.5707963267948966192313216916397514L);
#endif
		}();

		T re = real(other);
		T im = imag(other);
		T ux, vx;

		if (internal::isNaN(re) || internal::isNaN(im)) { // At least one NaN
			ux = internal::traits<T>::quietNaN();
			vx = ux;
		} else if (internal::isInf(re)) { // (+/-Inf, not NaN)
			ux = internal::copySign(T(0), re);
			vx = internal::copySign(piBy2, im);
		} else { // (finite, not NaN)
			const T magIm = ::librapid::abs(im);
			const T oldRe = re;

			re = ::librapid::abs(re);

			if (arcBig < re) { // |re| is large
				T fx = im / re;
				ux	 = 1 / re / (1 + fx * fx);
				vx	 = internal::copySign(piBy2, im);
			} else if (arcBig < magIm) { // |im| is large
				T fx = re / im;
				ux	 = fx / im / (1 + fx * fx);
				vx	 = internal::copySign(piBy2, im);
			} else if (re != 1) { // |re| is small
				T reFrom1 = 1 - re;
				T imEps2  = magIm * magIm;
				ux = T(0.25) * detail::algorithm::logP1(4 * re / (reFrom1 * reFrom1 + imEps2));
				vx = T(0.5) * ::librapid::atan2(2 * im, reFrom1 * (1 + re) - imEps2);
			} else if (im == 0) { // {+/-1, 0)
				ux = internal::traits<T>::infinity();
				vx = im;
			} else { // (+/-1, nonzero)
				ux = ::librapid::log(::librapid::sqrt(::librapid::sqrt(4 + im * im)) /
									 ::librapid::sqrt(magIm));
				vx = internal::copySign(T(0.5) * (piBy2 + ::librapid::atan2(magIm, T(2))), im);
			}
			ux = internal::copySign(ux, oldRe);
		}
		return Complex<T>(ux, vx);
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> atan(const Complex<T> &other) {
		Complex atanhVal = ::librapid::atanh(Complex<T>(-imag(other), real(other)));
		return Complex<T>(imag(atanhVal), -real(atanhVal));
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> cosh(const Complex<T> &other) {
		return Complex<T>(::librapid::cosh(real(other)) * ::librapid::cos(imag(other)),
						  ::librapid::sinh(real(other)) * ::librapid::sin(imag(other)));
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> polarPositiveNanInfZeroRho(const T &rho, const T &theta) {
		// Rho is +NaN/+Inf/+0
		if (internal::isNaN(theta) || internal::isInf(theta)) { // Theta is NaN/Inf
			if (internal::isInf(rho)) {
				return Complex<T>(rho, ::librapid::sin(theta)); // (Inf, NaN/Inf)
			} else {
				return Complex<T>(rho, internal::copySign(rho, theta)); // (NaN/0, NaN/Inf)
			}
		} else if (theta == T(0)) {		   // Theta is zero
			return Complex<T>(rho, theta); // (NaN/Inf/0, 0)
		} else {						   // Theta is finite non-zero
			// (NaN/Inf/0, finite non-zero)
			return Complex<T>(rho * ::librapid::cos(theta), rho * ::librapid::sin(theta));
		}
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> exp(const Complex<T> &other) {
		const T logRho = real(other);
		const T theta  = imag(other);

		if (!internal::isNaN(logRho) && !internal::isInf(logRho)) { // Real component is finite
			T real = logRho;
			T imag = logRho;
			detail::algorithm::expMul(&real, ::librapid::cos(theta), 0);
			detail::algorithm::expMul(&imag, ::librapid::sin(theta), 0);
			return Complex<T>(real, imag);
		}

		// Real component is NaN/Inf
		// Return polar(exp(re), im)
		if (internal::isInf(logRho)) {
			if (logRho < 0) {
				return polarPositiveNanInfZeroRho(T(0), theta); // exp(-Inf) = +0
			} else {
				return polarPositiveNanInfZeroRho(logRho, theta); // exp(+Inf) = +Inf
			}
		} else {
			return polarPositiveNanInfZeroRho(::librapid::abs(logRho), theta); // exp(NaN) = +NaN
		}
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> exp2(const Complex<T> &other) {
		return pow(T(2), other);
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> exp10(const Complex<T> &other) {
		return pow(T(10), other);
	}

	template<typename T>
	T _fabs(const Complex<T> &other, int64_t *exp) {
		*exp = 0;
		T av = ::librapid::abs(real(other));
		T bv = ::librapid::abs(imag(other));

		if (internal::isInf(av) || internal::isInf(bv)) {
			return internal::traits<T>::infinity(); // At least one component is Inf
		} else if (internal::isNaN(av)) {
			return av; // Real component is NaN
		} else if (internal::isNaN(bv)) {
			return bv; // Imaginary component is NaN
		} else {
			if (av < bv) std::swap(av, bv);
			if (av == 0) return av; // |0| = 0

			if (1 <= av) {
				*exp = 4;
				av	 = av * T(0.0625);
				bv	 = bv * T(0.0625);
			} else {
				const T fltEps	= internal::traits<T>::epsilon();
				const T legTiny = fltEps == 0 ? T(0) : 2 * internal::traits<T>::min() / fltEps;

				if (av < legTiny) {
#if defined(LIBRAPID_USE_MULTIPREC)
					int64_t exponent;
					if constexpr (std::is_same_v<T, mpfr>) {
						exponent = -2 * ::mpfr::mpreal::get_default_prec();
					} else {
						exponent = -2 * std::numeric_limits<T>::digits;
					}
#else
					exponent = -2 * std::numeric_limits<T>::digits;
#endif

					*exp = exponent;
					av	 = ::librapid::ldexp(av, -exponent);
					bv	 = ::librapid::ldexp(bv, -exponent);
				} else {
					*exp = -2;
					av	 = av * 4;
					bv	 = bv * 4;
				}
			}

			const T tmp = av - bv;
			if (tmp == av) {
				return av; // bv is unimportant
			} else {
#if defined(LIBRAPID_USE_MULTIPREC)
				if constexpr (std::is_same_v<T, mpfr>) { // No approximations
					const T root2		 = ::librapid::sqrt(mpfr(2));
					const T onePlusRoot2 = root2 + 1;

					const T qv = tmp / bv;
					const T rv = (qv + 2) * qv;
					const T sv = rv / (root2 + ::librapid::sqrt(rv + 2)) + onePlusRoot2 + qv;
					return av + bv / sv;
				} else {
#endif
					if (bv < tmp) { // Use a simple approximation
						const T qv = av / bv;
						return av + bv / (qv + ::librapid::sqrt(qv * qv + 1));
					} else { // Use 1 1/2 precision to preserve bits
						constexpr T root2 = static_cast<T>(1.4142135623730950488016887242096981L);
						constexpr T onePlusRoot2High = static_cast<T>(10125945.0 / 4194304.0);
						constexpr T onePlusRoot2Low =
						  static_cast<T>(1.4341252375973918872420969807856967e-7L);

						const T qv = tmp / bv;
						const T rv = (qv + 2) * qv;
						const T sv = rv / (root2 + ::librapid::sqrt(rv + 2)) + onePlusRoot2Low +
									 qv + onePlusRoot2High;
						return av + bv / sv;
					}
#if defined(LIBRAPID_USE_MULTIPREC)
				}
#endif
			}
		}
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T _logAbs(const Complex<T> &other) noexcept {
		return static_cast<T>(detail::algorithm::logHypot(static_cast<double>(real(other)),
														  static_cast<double>(imag(other))));
	}

#if defined(LIBRAPID_USE_MULTIPREC)
	template<>
	LR_NODISCARD("")
	LR_INLINE mpfr _logAbs(const Complex<mpfr> &other) noexcept {
		return detail::algorithm::logHypot(real(other), imag(other));
	}
#endif

	template<>
	LR_NODISCARD("")
	LR_INLINE float _logAbs(const Complex<float> &other) noexcept {
		return detail::algorithm::logHypot(real(other), imag(other));
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> log(const Complex<T> &other) {
		const T logAbs = _logAbs(other);
		const T theta  = ::librapid::atan2(imag(other), real(other));
		return Complex<T>(logAbs, theta);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> _pow(const T &left, const T &right) {
		if (0 <= left) {
			return Complex<T>(::librapid::pow(left, right), internal::copySign(T(0), right));
		} else {
			return exp(right * log(Complex<T>(left)));
		}
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> pow(const Complex<T> &left, const T &right) {
		if (imag(left) == 0) {
			if (internal::signBit(imag(left))) {
				return conj(_pow(real(left), right));
			} else {
				return _pow(real(left), right);
			}
		} else {
			return exp(right * log(left));
		}
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> pow(const T &left, const Complex<T> &right) {
		if (imag(right) == 0) {
			return _pow(left, real(right));
		} else if (0 < left) {
			return exp(right * ::librapid::log(left));
		} else {
			return exp(right * log(Complex<T>(left)));
		}
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> pow(const Complex<T> &left, const Complex<T> &right) {
		if (imag(right) == 0) {
			return pow(left, real(right));
		} else if (imag(left) == 0 && 0 < real(left)) {
			return exp(right * ::librapid::log(real(left)));
		} else {
			return exp(right * log(left));
		}
	}

	// Return sinh(left) * right
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LR_NODISCARD("")
	LR_INLINE T _sinh(const T left, const T right) {
		return static_cast<T>(::librapid::sinh(static_cast<double>(left)) *
							  static_cast<double>(right));
	}

	template<typename T, typename std::enable_if_t<!std::is_fundamental_v<T>, int> = 0>
	LR_NODISCARD("")
	LR_INLINE T _sinh(const T &left, const T &right) {
		return ::librapid::sinh(left) * right;
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> sinh(const Complex<T> &other) {
		return Complex<T>(::librapid::sinh(real(other)) * ::librapid::cos(imag(other)),
						  ::librapid::cosh(real(other)) * ::librapid::sin(imag(other)));
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> sqrt(const Complex<T> &other) {
		int64_t otherExp;
		T rho = _fabs(other, &otherExp); // Get magnitude and scale factor

		if (otherExp == 0) { // Argument is zero, Inf or NaN
			if (rho == 0) {
				return Complex<T>(T(0), imag(other));
			} else if (internal::isInf(rho)) {
				const T re = real(other);
				const T im = imag(other);

				if (internal::isInf(im)) {
					return Complex<T>(internal::traits<T>::infinity(), im); // (any, +/-Inf)
				} else if (internal::isNaN(im)) {
					if (re < 0) {
						// (-Inf, NaN)
						return Complex<T>(::librapid::abs(im), internal::copySign(re, im));
					} else {
						return other; // (+Inf, NaN)
					}
				} else {
					if (re < 0) {
						return Complex<T>(T(0), internal::copySign(re, im)); // (-Inf, finite)
					} else {
						return Complex<T>(re, internal::copySign(T(0), im)); // (+Inf, finite)
					}
				}
			} else {
				return Complex<T>(rho, rho);
			}
		} else { // Compute in safest quadrant
			T realMag = internal::ldexp(::librapid::abs(real(other)), -otherExp);
			rho		  = internal::ldexp(::librapid::sqrt(2 * (realMag + rho)), otherExp / 2 - 1);
			if (0 <= real(other)) {
				return Complex<T>(rho, imag(other) / (2 * rho));
			} else {
				return Complex<T>(::librapid::abs(imag(other) / (2 * rho)),
								  internal::copySign(rho, imag(other)));
			}
		}
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> tanh(const Complex<T> &other) {
		T tv = ::librapid::tan(imag(other));
		T sv = ::librapid::sinh(real(other));
		T bv = sv * (T(1) + tv * tv);
		T dv = T(1) + bv * sv;

		if (internal::isInf(dv)) {
			T real;
			if (sv < T(0))
				real = T(-1);
			else
				real = T(1);
			return Complex<T>(real, T(0));
		}
		return Complex<T>((::librapid::sqrt(T(1) + sv * sv)) * bv / dv, tv / dv);
	}

	// Return the phase angle of a complex value as a real
	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T arg(const Complex<T> &other) {
		return ::librapid::atan2(imag(other), real(other));
	}

	// Return complex projection
	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> proj(const Complex<T> &other) {
		if (internal::isInf(real(other)) || internal::isInf(imag(other))) {
			const T im = internal::copySign(T(0), imag(other));
			return Complex<T>(internal::traits<T>::infinity(), im);
		}
		return other;
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> cos(const Complex<T> &other) {
		return Complex<T>(::librapid::cosh(imag(other)) * ::librapid::cos(real(other)),
						  -::librapid::sinh(imag(other)) * ::librapid::sin(real(other)));
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> csc(const Complex<T> &other) {
		return T(1) / sin(other);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> sec(const Complex<T> &other) {
		return T(1) / cos(other);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> cot(const Complex<T> &other) {
		return T(1) / tan(other);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> acsc(const Complex<T> &other) {
		return asin(T(1) / other);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> asec(const Complex<T> &other) {
		return acos(T(1) / other);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> acot(const Complex<T> &other) {
		return atan(T(1) / other);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> log2(const Complex<T> &other) {
		return log(other) / ::librapid::log(T(2));
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> log10(const Complex<T> &other) {
		return log(other) / ::librapid::log(T(10.0));
	}

	// Return magnitude squared
	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T norm(const Complex<T> &other) {
		return real(other) * real(other) + imag(other) * imag(other);
	}

	// Return rho * exp(i * theta);
	template<typename T>
	LR_NODISCARD("")
	Complex<T> polar(const T &rho, const T &theta) {
		if (!internal::isNaN(rho) && !internal::isInf(rho) && rho != T(0)) {
			// Rho is finite and non-zero
			return Complex<T>(rho * ::librapid::cos(theta), rho * ::librapid::sin(theta));
		}

		// Rho is NaN/Inf/0
		if (internal::signBit(rho))
			return -polarPositiveNanInfZeroRho(-rho, theta);
		else
			return polarPositiveNanInfZeroRho(rho, theta);
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> atan2(const Complex<T> &y, const Complex<T> &x) {
		LR_ASSERT(false, "Complex atan2 is not yet implemented");
		return {0};
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> sin(const Complex<T> &other) {
		return Complex<T>(::librapid::cosh(imag(other)) * ::librapid::sin(real(other)),
						  ::librapid::sinh(imag(other)) * ::librapid::cos(real(other)));
	}

	template<typename T>
	LR_NODISCARD("")
	Complex<T> tan(const Complex<T> &other) {
		Complex<T> zv(tanh(Complex<T>(-imag(other), real(other))));
		return Complex<T>(imag(zv), -real(zv));
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> floor(const Complex<T> &other) {
		return Complex<T>(::librapid::floor(real(other)), ::librapid::floor(imag(other)));
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> ceil(const Complex<T> &other) {
		return Complex<T>(::librapid::ceil(real(other)), ::librapid::ceil(imag(other)));
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE Complex<T> random(const Complex<T> &min, const Complex<T> &max, uint64_t seed = -1) {
		return Complex<T>(::librapid::random(real(min), real(max), seed),
						  ::librapid::random(imag(min), imag(max), seed));
	}

	template<typename T>
	LR_NODISCARD("")
	std::string str(const Complex<T> &other, const StrOpt &options = DEFAULT_STR_OPT) {
		if (!internal::signBit(imag(other)))
			return "(" + str(real(other), options) + "+" + str(imag(other), options) + "i)";
		else
			return "(" + str(real(other), options) + "-" + str(-imag(other), options) + "i)";
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
			bool signBit	 = librapid::internal::signBit(librapid::imag(num));
			std::string real = fmt::format(formatStr, num.real());
			std::string imag = fmt::format(formatStr, (signBit ? -num.imag() : num.imag()));
			std::string ret;
			if (!signBit)
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