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

namespace librapid { namespace detail {
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
		LR_INLINE constexpr Fmp<T> addSmallX2(const T &x, const T &y) noexcept {
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
		LR_INLINE constexpr Fmp<T> addSmallX2(const Fmp<T> &x, const Fmp<T> &y) noexcept {
			const Fmp<T> sum0 = addSmallX2(x, y.val0);
			return addSmallX2(sum0.val0, sum0.val1 + y.val1);
		}

		// 2x precision + 2x precision -> 1x precision
		template<typename T>
		LR_NODISCARD("")
		LR_INLINE constexpr T addX1(const Fmp<T> &x, const Fmp<T> &y) noexcept {
			const Fmp<T> sum0 = addx2(x.val0, y.val0);
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

		// square(1x precision) -> 2x precision
		// the result is exact when no internal overflow or underflow occurs
		LR_NODISCARD("") LR_INLINE Fmp<double> sqrX2(const double x) noexcept {
			const double prod0 = x * x;
			return {prod0, sqrError(x, prod0)};
		}
	} // namespace multiprec

	namespace algorithm {
		// HypotLegHuge = T{0.5} * sqrt((numeric_limits<T>::max()));
		// HypotLegTiny = sqrt(T{2.0} * (numeric_limits<T>::min)() / numeric_limits<T>::epsilon());

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
			static inline T val =
			  ::librapid::sqrt(T(2) * internal::traits<T>::min()) / internal::traits<T>::epsilon();
		};

		template<>
		struct HypotLegTinyHelper<double> {
			static constexpr double val = 1.4156865331029228e-146;
		};

		template<>
		struct HypotLegTinyHelper<float> {
			static constexpr double val = 4.440892e-16f;
		};
	} // namespace algorithm
}}	  // namespace librapid::detail

#undef USE_X86_X64_INTRINSICS
#undef USE_ARM64_INTRINSICS