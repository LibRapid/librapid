#ifndef LIBRAPID_MATH_COMPLEX_HPP
#define LIBRAPID_MATH_COMPLEX_HPP

/*
 * A Complex Number implementation, based off of MSVC's std::complex<T> datatype. This type does
 * not conform to the C++ standard, but its a wider range of primitive and user-defined types.
 * Furthermore, it integrates much more nicely with the rest of LibRapid and provides a lot more
 * functionality.
 *
 * See below for the MSVC implementation :)
 * https://github.com/microsoft/STL/blob/main/stl/inc/complex
 *
 */

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

			/// \brief Summarizes two 1x precision values combined into a 2x precision result
			///
			/// This function is exact when:
			/// 1. The result doesn't overflow
			/// 2. Either underflow is gradual, or no internal underflow occurs
			/// 3. Intermediate precision is either the same as T, or greater than twice the
			/// precision of T
			/// 4. Parameters and local variables do not retain extra intermediate precision
			/// 5. Rounding mode is rounding to nearest.
			///
			/// Violation of condition 3 or 5 could lead to relative error on the order of
			/// epsilon^2.
			///
			/// Violation of other conditions could lead to worse results
			///
			/// \tparam T Template type
			/// \param x First value
			/// \param y Second value
			/// \return Sum of x and y
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr Fmp<T> addX2(const T &x,
																			 const T &y) noexcept {
				const T sum0 = x + y;
				const T yMod = sum0 - x;
				const T xMod = sum0 - yMod;
				const T yErr = y - yMod;
				const T xErr = x - xMod;
				return {sum0, xErr + yErr};
			}

			/// \brief Combines two 1x precision values into a 2x precision result with the
			/// requirement of specific exponent relationship
			///
			/// Requires: exponent(x) + countr_zero(significand(x)) >= exponent(y) or x == 0
			///
			/// The result is exact when:
			/// 1. The requirement above is satisfied
			/// 2. No internal overflow occurs
			/// 3. Either underflow is gradual, or no internal underflow occurs
			/// 4. Intermediate precision is either the same as T, or greater than twice the
			///    precision of T
			/// 5. Parameters and local variables do not retain extra intermediate precision
			/// 6. Rounding mode is rounding to nearest
			///
			/// Violation of condition 3 or 5 could lead to relative error on the order of
			/// epsilon^2.
			///
			/// Violation of other conditions could lead to worse results
			///
			/// \tparam T Template type
			/// \param x First value
			/// \param y Second value
			/// \return Sum of x and y
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr Fmp<T>
			addSmallX2(const T x, const T y) noexcept {
				const T sum0 = x + y;
				const T yMod = sum0 - x;
				const T yErr = y - yMod;
				return {sum0, yErr};
			}

			/// \brief Combines a 1x precision value with a 2x precision value
			///
			/// Requires: exponent(x) + countr_zero(significand(x)) >= exponent(y.val0) or x == 0
			///
			/// \tparam T Template type
			/// \param x First value
			/// \param y Second value
			/// \return Sum of x and y
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr Fmp<T>
			addSmallX2(const T &x, const Fmp<T> &y) noexcept {
				const Fmp<T> sum0 = addSmallX2(x, y.val0);
				return addSmallX2(sum0.val0, sum0.val1 + y.val1);
			}

			/// \brief Combines two 2x precision values into a 1x precision result
			/// \tparam T Template type
			/// \param x First value
			/// \param y Second value
			/// \return Sum of x and y
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr T addX1(const Fmp<T> &x,
																		const Fmp<T> &y) noexcept {
				const Fmp<T> sum0 = addX2(x.val0, y.val0);
				return sum0.val0 + (sum0.val1 + (x.val1 + y.val1));
			}

			/// \brief Rounds a 2x precision value to 26 significant bits
			/// \param x Value to round
			/// \return Rounded value
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr double
			highHalf(const double x) noexcept {
				const auto bits			= bitCast<uint64_t>(x);
				const auto highHalfBits = (bits + 0x3ff'ffffULL) & 0xffff'ffff'f800'0000ULL;
				return bitCast<double>(highHalfBits);
			}

#if defined(USE_X86_X64_INTRINSICS) || defined(USE_ARM64_INTRINSICS) // SIMD method
			/// \brief Calculates the error between x^2 and its faithfully rounded product prod0
			///
			/// The result is exact when:
			/// 1. prod0 is x^2 faithfully rounded
			/// 2. No internal overflow or underflow occurs
			///
			/// Violation of condition 1 could lead to relative error on the order of epsilon.
			///
			/// \param x Input value
			/// \param prod0 Faithfully rounded product of x^2
			/// \return Error between x^2 and prod0
			LIBRAPID_NODISCARD
			LIBRAPID_ALWAYS_INLINE double sqrError(const double x, const double prod0) noexcept {
#	if defined(USE_X86_X64_INTRINSICS)
				const __m128d xVec		= _mm_set_sd(x);
				const __m128d prodVec	= _mm_set_sd(prod0);
				const __m128d resultVec = _mm_fmsub_sd(xVec, xVec, prodVec);
				double result;
				_mm_store_sd(&result, resultVec);
				return result;
#	else // Only two options, so this is fine
				const float64x1_t xVec		= vld1_double(&x);
				const float64x1_t prod0Vec	= vld1_double(&prod0);
				const float64x1_t resultVec = vfma_double(vneg_double(prod0Vec), xVec, xVec);
				double result;
				vst1_double(&result, resultVec);
				return result;
#	endif
			}
#else
			/// \brief Fallback method for sqrError(const double, const double) when SIMD is not
			/// available.
			LIBRAPID_NODISCARD
			LIBRAPID_ALWAYS_INLINE constexpr double sqrError(const double x,
															 const double prod0) noexcept {
				const double xHigh = highHalf(x);
				const double xLow  = x - xHigh;
				return ((xHigh * xHigh - prod0) + 2.0 * xHigh * xLow) + xLow * xLow;
			}
#endif

			/// \brief Type-agnostic version of sqrError(const double, const double)
			/// \tparam T Template type
			/// \param x Input value
			/// \param prod0 Faithfully rounded product of x^2
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T sqrError(const T x,
																 const T prod0) noexcept {
				const T xHigh = static_cast<T>(highHalf(x));
				const T xLow  = x - xHigh;
				return ((xHigh * xHigh - prod0) + static_cast<T>(2.0) * xHigh * xLow) + xLow * xLow;
			}

			/// \brief Calculates the square of a 1x precision value and returns a 2x precision
			/// result
			///
			/// The result is exact when no internal overflow or underflow occurs.
			///
			/// \param x Input value
			/// \return 2x precision square of x
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Fmp<double> sqrX2(const double x) noexcept {
				const double prod0 = x * x;
				return {prod0, sqrError(x, prod0)};
			}

			/// \brief Type-agnostic version of sqrX2(const double)
			/// \tparam T Template type
			/// \param x Input value
			/// \return 2x precision square of x
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Fmp<T> sqrX2(const T x) noexcept {
				const T prod0 = x * x;
				return {prod0, static_cast<T>(sqrError(x, prod0))};
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
				static inline T val =
				  (std::is_integral_v<T>)
					? (::librapid::sqrt(typetraits::TypeInfo<T>::max()) / T(2))
					: (T(0.5) * ::librapid::sqrt(typetraits::TypeInfo<T>::max()));
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
				static inline T val = ::librapid::sqrt(T(2) * typetraits::TypeInfo<T>::min() /
													   typetraits::TypeInfo<T>::epsilon());
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

			/// \brief Calculates \f$ x^2 + y^2 - 1 \f$ for \f$ |x| \geq |y| \f$ and \f$ 0.5 \leq
			/// |x| < 2^{12} \f$ \tparam T Template type \param x First value \param y Second value
			/// \return x * x + y * y - 1
			template<typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T normMinusOne(const T x,
																	 const T y) noexcept {
				const multiprec::Fmp<T> xSqr   = multiprec::sqrX2(x);
				const multiprec::Fmp<T> ySqr   = multiprec::sqrX2(y);
				const multiprec::Fmp<T> xSqrM1 = multiprec::addSmallX2(T(-1), xSqr);
				return multiprec::addX1(xSqrM1, ySqr);
			}

			/// \brief Calculates \f$ \log(1 + x) \f$
			///
			/// May be inaccurate for small inputs
			///
			/// \tparam safe If true, will check for NaNs and overflow
			/// \tparam T Template type
			/// \param x Input value
			/// \return \f$ \log(1 + x) \f$
			template<bool safe = true, typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T logP1(const T x) {
				if constexpr (!safe) return ::librapid::log(x + 1.0);
#if defined(LIBRAPID_USE_MULTIPREC)
				// No point doing anything shown below if we're using multiprec
				if constexpr (std::is_same_v<T, mpfr>) return ::librapid::log(x + 1.0);
#endif

				if (::librapid::isNaN(x)) return x + x; // Trigger a signaling NaN

				// Naive formula
				if (x <= T(-0.5) || T(2) <= x) {
					// To avoid overflow
					if (x == typetraits::TypeInfo<T>::max()) return ::librapid::log(x);
					return ::librapid::log(T(1) + x);
				}

				const T absX = ::librapid::abs(x);
				if (absX < typetraits::TypeInfo<T>::epsilon()) {
					if (x == T(0)) return x;
					return x - T(0.5) * x * x; // Honour rounding
				}

				// log(1 + x) with fix for small x
				const multiprec::Fmp<T> tmp = multiprec::addSmallX2(T(1), x);
				return ::librapid::log(tmp.val0) + tmp.val1 / tmp.val0;
			}

			// Return log(hypot(x, y))

			/// \brief Calculates \f$ \log(\sqrt{x^2 + y^2}) \f$
			/// \tparam safe If true, will check for NaNs and overflow
			/// \tparam T Template type
			/// \param x Horizontal component
			/// \param y Vertical component
			/// \return \f$ \log(\sqrt{x^2 + y^2}) \f$
			template<bool safe = true, typename T>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T logHypot(const T x, const T y) noexcept {
				if constexpr (!safe) return ::librapid::log(::librapid::sqrt(x * x + y * y));
#if defined(LIBRAPID_USE_MULTIPREC)
				// No point doing anything shown below if we're using multiprec
				if constexpr (std::is_same_v<T, mpfr>)
					return ::librapid::log(::mpfr::hypot(x, y));
				else {
#endif

					if (!::librapid::isFinite(x) || !::librapid::isFinite(y)) { // Inf or NaN
						// Return NaN and raise FE_INVALID if either x or y is NaN
						if (::librapid::isNaN(x) || ::librapid::isNaN(y)) return x + y;

						// Return Inf if either of them is infinity
						if (::librapid::isInf(x)) return x;
						if (::librapid::isInf(y)) return y;

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

			/// \brief Compute \f$e^{\text{pleft}} \times \text{right} \times 2^{\text{exponent}}\f$
			///
			/// \tparam T Template type
			/// \param pleft Pointer to the value to be exponentiated
			/// \param right Multiplier for the exponentiated value
			/// \param exponent Exponent for the power of 2 multiplication
			/// \return 1 if the result is NaN or Inf, -1 otherwise
			template<typename T>
			short expMul(T *pleft, T right, short exponent) {
#if defined(LIBRAPID_USE_MULTIPREC)
				if constexpr (std::is_same_v<T, mpfr>) {
					*pleft = ::mpfr::exp(*pleft) * right * ::mpfr::exp2(exponent);
					return (::librapid::isNaN(*pleft) || ::librapid::isInf(*pleft)) ? 1 : -1;
				} else {
#endif

#if defined(LIBRAPID_MSVC)
					auto tmp  = static_cast<double>(*pleft);
					short ans = _CSTD _Exp(&tmp, static_cast<double>(right), exponent);
					*pleft	  = static_cast<T>(tmp);
					return ans;
#else
				*pleft = ::librapid::exp(*pleft) * right * ::librapid::exp2(exponent);
				return (::librapid::isNaN(*pleft) || ::librapid::isInf(*pleft)) ? 1 : -1;
#endif

#if defined(LIBRAPID_USE_MULTIPREC)
				} // This ensures the "if constexpr" above actually stops compiler errors
#endif
			}
		} // namespace algorithm
	}	  // namespace detail

	/// \brief A class representing a complex number of the form \f$a + bi\f$, where \f$a\f$ and
	///        \f$b\f$ are real numbers
	///
	/// This class represents a complex number of the form \f$a + bi\f$, where \f$a\f$ and
	/// \f$b\f$ are real numbers. The class is templated, allowing the user to specify the type
	/// of the real and imaginary components. The default type is ``double``.
	///
	/// \tparam T The type of the real and imaginary components
	template<typename T = double>
	class Complex {
	public:
		using Scalar = typename typetraits::TypeInfo<T>::Scalar;

		/// \brief Default constructor
		///
		/// Create a new complex number. Both the real and imaginary components are set to zero
		Complex() : m_val {T(0), T(0)} {}

		/// \brief Construct a complex number from a real number
		///
		/// Create a complex number, setting only the real component. The imaginary component is
		/// initialized to zero
		///
		/// \tparam R The type of the real component
		/// \param realVal The real component
		template<typename R>
		explicit Complex(const R &realVal) : m_val {T(realVal), T(0)} {}

		/// \brief Construct a complex number from real and imaginary components
		///
		/// Create a new complex number where both the real and imaginary parts are set from the
		/// passed parameters
		///
		/// \tparam R The type of the real component
		/// \tparam I The type of the imaginary component
		/// \param realVal The real component
		/// \param imagVal The imaginary component
		template<typename R, typename I>
		Complex(const R &realVal, const I &imagVal) : m_val {T(realVal), T(imagVal)} {}

		/// \brief Complex number copy constructor
		/// \param other The complex number to copy
		Complex(const Complex<T> &other) : m_val {other.real(), other.imag()} {}

		/// \brief Complex number move constructor
		/// \param other The complex number to move
		Complex(Complex<T> &&other) noexcept : m_val {other.real(), other.imag()} {}

		/// \brief Construct a complex number from another complex number with a different type
		/// \tparam Other Type of the components of the other complex number
		/// \param other The complex number to copy
		template<typename Other>
		Complex(const Complex<Other> &other) : m_val {T(other.real()), T(other.imag())} {}

		/// \brief Construct a complex number from a std::complex
		/// \param other The std::complex value to copy
		explicit Complex(const std::complex<T> &other) : m_val {other.real(), other.imag()} {}

		static constexpr size_t size() { return typetraits::TypeInfo<Complex>::packetWidth; }

		/// \brief Complex number assignment operator
		/// \param other The value to assign
		/// \return *this
		Complex<T> &operator=(const Complex<T> &other) {
			if (this == &other) return *this;
			m_val[RE] = other.real();
			m_val[IM] = other.imag();
			return *this;
		}

		template<typename P>
		LIBRAPID_ALWAYS_INLINE void store(P *ptr) const {
			auto casted = reinterpret_cast<Scalar *>(ptr);
			auto ret	= Vc::interleave(m_val[RE], m_val[IM]);
			ret.first.store(casted);
			ret.second.store(casted + size());
		}

		template<typename P>
		LIBRAPID_ALWAYS_INLINE void load(const P *ptr) {
			auto casted = reinterpret_cast<const Scalar *>(ptr);
			Vc::deinterleave(&m_val[RE], &m_val[IM], casted, Vc::Aligned);
		}

		/// \brief Assign to the real component
		///
		/// Set the real component of this complex number to \p val
		///
		/// \param val The value to assign
		LIBRAPID_ALWAYS_INLINE void real(const T &val) { m_val[RE] = val; }

		/// \brief Assign to the imaginary component
		///
		/// Set the imaginary component of this complex number to \p val
		///
		/// \param val The value to assign
		LIBRAPID_ALWAYS_INLINE void imag(const T &val) { m_val[IM] = val; }

		/// \brief Access the real component
		///
		/// Returns a const reference to the real component of this complex number
		///
		/// \return Real component
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const T &real() const { return m_val[RE]; }

		/// \brief Access the imaginary component
		///
		/// Returns a const reference to the imaginary component of this complex number
		///
		/// \return Imaginary component
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const T &imag() const { return m_val[IM]; }

		/// \brief Access the real component
		///
		/// Returns a reference to the real component of this complex number. Since this is a
		/// reference type, it can be assigned to
		///
		/// \return Real component
		LIBRAPID_ALWAYS_INLINE T &real() { return m_val[RE]; }

		/// \brief Access the imaginary component
		///
		/// Returns a reference to the imaginary component of this complex number. Since this is a
		/// reference type, it can be assigned to
		///
		/// \return imaginary component
		LIBRAPID_ALWAYS_INLINE T &imag() { return m_val[IM]; }

		/// \brief Complex number assigment operator
		///
		/// Set the real component of this complex number to \p other, and the imaginary component
		/// to 0
		///
		/// \param other
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator=(const T &other) {
			m_val[RE] = other;
			m_val[IM] = 0;
			return *this;
		}

		/// \brief Complex number assigment operator
		///
		/// Assign another complex number to this one, copying the real and imaginary components
		///
		/// \tparam Other The type of the other complex number
		/// \param other Complex number to assign
		/// \return *this
		template<typename Other>
		LIBRAPID_ALWAYS_INLINE Complex &operator=(const Complex<Other> &other) {
			m_val[RE] = static_cast<T>(other.real());
			m_val[IM] = static_cast<T>(other.real());
			return *this;
		}

		/// \brief Inplace addition
		///
		/// Add a scalar value to the real component of this imaginary number
		///
		/// \param other Scalar value to add
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator+=(const T &other) {
			m_val[RE] = m_val[RE] + other;
			return *this;
		}

		/// \brief Inplace subtraction
		///
		/// Subtract a scalar value from the real component of this imaginary number
		///
		/// \param other Scalar value to subtract
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator-=(const T &other) {
			m_val[RE] = m_val[RE] - other;
			return *this;
		}

		/// \brief Inplace multiplication
		///
		/// Multiply both the real and imaginary components of this complex number by a scalar
		///
		/// \param other Scalar value to multiply by
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator*=(const T &other) {
			m_val[RE] = m_val[RE] * other;
			m_val[IM] = m_val[IM] * other;
			return *this;
		}

		/// \brief Inplace division
		///
		/// Divide both the real and imaginary components of this complex number by a scalar
		///
		/// \param other Scalar value to divide by
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator/=(const T &other) {
			m_val[RE] = m_val[RE] / other;
			m_val[IM] = m_val[IM] / other;
			return *this;
		}

		/// \brief Inplace addition
		///
		/// Add a complex number to this one
		///
		/// \param other Complex number to add
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator+=(const Complex &other) {
			this->_add(other);
			return *this;
		}

		/// \brief Inplace subtraction
		///
		/// Subtract a complex number from this one
		///
		/// \param other Complex number to subtract
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator-=(const Complex &other) {
			this->_sub(other);
			return *this;
		}

		/// \brief Inplace multiplication
		///
		/// Multiply this complex number by another one
		///
		/// \param other Complex number to multiply by
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator*=(const Complex &other) {
			this->_mul(other);
			return *this;
		}

		/// \brief Inplace division
		///
		/// Divide this complex number by another one
		///
		/// \param other Complex number to divide by
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Complex &operator/=(const Complex &other) {
			this->_div(other);
			return *this;
		}

		/// \brief Cast to scalar types
		///
		/// Cast this complex number to a scalar type. This will extract only the real component.
		///
		/// \tparam To Type to cast to
		/// \return Scalar
		template<typename To>
		LIBRAPID_ALWAYS_INLINE explicit operator To() const {
			return static_cast<To>(m_val[RE]);
		}

		/// \brief Cast to a complex number with a different scalar type
		///
		/// Cast the real and imaginary components of this complex number to a different type and
		/// return the result as a new complex number
		///
		/// \tparam To Scalar type to cast to
		/// \return Complex number
		template<typename To>
		LIBRAPID_ALWAYS_INLINE explicit operator Complex<To>() const {
			return Complex<To>(static_cast<To>(m_val[RE]), static_cast<To>(m_val[IM]));
		}

		/// \brief Complex number to string
		///
		/// Create a std::string representation of a complex number, formatting each component with
		/// the format string
		///
		/// \param format Format string
		/// \return std::string
		LIBRAPID_NODISCARD std::string str(const std::string &format = "{}") const {
			if (!::librapid::signBit(m_val[IM]))
				return "(" + fmt::format(format, m_val[RE]) + "+" + fmt::format(format, m_val[IM]) +
					   "j)";
			else
				return "(" + fmt::format(format, m_val[RE]) + "-" +
					   fmt::format(format, -m_val[IM]) + "j)";
		}

	protected:
		/// \brief Add a complex number to this one
		/// \tparam Other Scalar type of the other complex number
		/// \param other Other complex number
		template<typename Other>
		LIBRAPID_ALWAYS_INLINE void _add(const Complex<Other> &other) {
			m_val[RE] = m_val[RE] + other.real();
			m_val[IM] = m_val[IM] + other.imag();
		}

		/// \brief Subtract a complex number from this one
		/// \tparam Other Scalar type of the other complex number
		/// \param other Other complex number
		template<typename Other>
		LIBRAPID_ALWAYS_INLINE void _sub(const Complex<Other> &other) {
			m_val[RE] = m_val[RE] - other.real();
			m_val[IM] = m_val[IM] - other.imag();
		}

		/// \brief Multiply this complex number by another one
		/// \tparam Other Scalar type of the other complex number
		/// \param other Other complex number
		template<typename Other>
		LIBRAPID_ALWAYS_INLINE void _mul(const Complex<Other> &other) {
			T otherReal = static_cast<T>(other.real());
			T otherImag = static_cast<T>(other.imag());

			T tmp	  = m_val[RE] * otherReal - m_val[IM] * otherImag;
			m_val[IM] = m_val[RE] * otherImag + m_val[IM] * otherReal;
			m_val[RE] = tmp;
		}

		/// \brief Divide this complex number by another one
		/// \tparam Other Scalar type of the other complex number
		/// \param other Other complex number
		template<typename Other>
		LIBRAPID_ALWAYS_INLINE void _div(const Complex<Other> &other) {
			T otherReal = static_cast<T>(other.real());
			T otherImag = static_cast<T>(other.imag());

			if (::librapid::isNaN(otherReal) || ::librapid::isNaN(otherImag)) { // Set result to NaN
				m_val[RE] = typetraits::TypeInfo<T>::quietNaN();
				m_val[IM] = m_val[RE];
			} else if ((otherImag < 0 ? T(-otherImag)
									  : T(+otherImag)) < // |other.imag()| < |other.real()|
					   (otherReal < 0 ? T(-otherReal) : T(+otherReal))) {
				T wr = otherImag / otherReal;
				T wd = otherReal + wr * otherImag;

				if (::librapid::isNaN(wd) || wd == 0) { // NaN result
					m_val[RE] = typetraits::TypeInfo<T>::quietNaN();
					m_val[IM] = m_val[RE];
				} else { // Valid result
					T tmp	  = (m_val[RE] + m_val[IM] * wr) / wd;
					m_val[IM] = (m_val[IM] - m_val[RE] * wr) / wd;
					m_val[RE] = tmp;
				}
			} else if (otherImag == 0) { // Set NaN
				m_val[RE] = typetraits::TypeInfo<T>::quietNaN();
				m_val[IM] = m_val[RE];
			} else { // 0 < |other.real()| <= |other.imag()|
				T wr = otherReal / otherImag;
				T wd = otherImag + wr * otherReal;

				if (::librapid::isNaN(wd) || wd == 0) { // NaN result
					m_val[RE] = typetraits::TypeInfo<T>::quietNaN();
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

	/// \brief Negate a complex number
	/// \tparam T Scalar type of the complex number
	/// \param other Complex number to negate
	/// \return Negated complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(const Complex<T> &other) {
		return Complex<T>(-other.real(), -other.imag());
	}

	/// \brief Add two complex numbers
	///
	/// Add two complex numbers together, returning the result
	///
	/// \tparam L Scalar type of LHS
	/// \tparam R Scalar type of RHS
	/// \param left LHS complex number
	/// \param right RHS complex number
	/// \return Sum of LHS and RHS
	template<typename L, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+(const Complex<L> &left,
															 const Complex<R> &right) {
		using Scalar = typename std::common_type_t<L, R>;
		Complex<Scalar> tmp(left.real(), left.imag());
		tmp += Complex<Scalar>(right.real(), right.imag());
		return tmp;
	}

	/// \brief Add a complex number and a scalar
	///
	/// Add a real number to the real component of a complex number, returning the result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS complex number
	/// \param right RHS scalar
	/// \return Sum of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+(const Complex<T> &left,
															 const R &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() + right);
		return tmp;
	}

	/// \brief Add a scalar to a complex number
	///
	/// Add a real number to the real component of a complex number, returning the result
	///
	/// \tparam R Type of the real number
	/// \tparam T Scalar type of the complex number
	/// \param left LHS scalar
	/// \param right RHS complex number
	/// \return Sum of LHS and RHS
	template<typename R, typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+(const R &left,
															 const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp += right;
		return tmp;
	}

	/// \brief Subtract a complex number from another complex number
	///
	/// Subtract the real and imaginary components of the RHS complex number from the corresponding
	/// components of the LHS complex number, returning the result
	///
	/// \tparam L Scalar type of the LHS complex number
	/// \tparam R Scalar type of the RHS complex number
	/// \param left LHS complex number
	/// \param right RHS complex number
	/// \return Difference of LHS and RHS
	template<typename L, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(const Complex<L> &left,
															 const Complex<R> &right) {
		using Scalar = typename std::common_type_t<L, R>;
		Complex<Scalar> tmp(left.real(), left.imag());
		tmp -= Complex<Scalar>(right.real(), right.imag());
		return tmp;
	}

	/// \brief Subtract a scalar from a complex number
	///
	/// Subtract a real number from the real component of a complex number, returning the result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS complex number
	/// \param right RHS scalar
	/// \return Difference of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(const Complex<T> &left,
															 const R &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() - right);
		return tmp;
	}

	/// \brief Subtract a complex number from a scalar
	///
	/// Subtract the real and imaginary components of the RHS complex number from a real number,
	/// returning the result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS scalar
	/// \param right RHS complex number
	/// \return Difference of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(const R &left,
															 const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp -= right;
		return tmp;
	}

	/// \brief Multiply two complex numbers
	///
	/// Multiply the LHS and RHS complex numbers, returning the result
	///
	/// \tparam L Scalar type of the LHS complex number
	/// \tparam R Scalar type of the RHS complex number
	/// \param left LHS complex number
	/// \param right RHS complex number
	/// \return Product of LHS and RHS
	template<typename L, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*(const Complex<L> &left,
															 const Complex<R> &right) {
		using Scalar = typename std::common_type_t<L, R>;
		Complex<Scalar> tmp(left.real(), left.imag());
		tmp *= Complex<Scalar>(right.real(), right.imag());
		return tmp;
	}

	/// \brief Multiply a complex number by a scalar
	///
	/// Multiply the real and imaginary components of a complex number by a real number, returning
	/// the result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS complex number
	/// \param right RHS scalar
	/// \return Product of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*(const Complex<T> &left,
															 const R &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() * right);
		tmp.imag(tmp.imag() * right);
		return tmp;
	}

	/// \brief Multiply a scalar by a complex number
	///
	/// Multiply a real number by the real and imaginary components of a complex number, returning
	/// the result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS scalar
	/// \param right RHS complex number
	/// \return Product of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*(const R &left,
															 const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp *= right;
		return tmp;
	}

	/// \brief Divide two complex numbers
	///
	/// Divide the LHS complex number by the RHS complex number, returning the result
	///
	/// \tparam L Scalar type of the LHS complex number
	/// \tparam R Scalar type of the RHS complex number
	/// \param left LHS complex number
	/// \param right RHS complex number
	/// \return Quotient of LHS and RHS
	template<typename L, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/(const Complex<L> &left,
															 const Complex<R> &right) {
		using Scalar = typename std::common_type_t<L, R>;
		Complex<Scalar> tmp(left.real(), left.imag());
		tmp /= Complex<Scalar>(right.real(), right.imag());
		return tmp;
	}

	/// \brief Divide a complex number by a scalar
	///
	/// Divide the real and imaginary components of a complex number by a real number, returning the
	/// result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS complex number
	/// \param right RHS scalar
	/// \return Quotient of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/(const Complex<T> &left,
															 const R &right) {
		Complex<T> tmp(left);
		tmp.real(tmp.real() / right);
		tmp.imag(tmp.imag() / right);
		return tmp;
	}

	/// \brief Divide a scalar by a complex number
	///
	/// Divide a real number by the real and imaginary components of a complex number, returning the
	/// result
	///
	/// \tparam T Scalar type of the complex number
	/// \tparam R Type of the real number
	/// \param left LHS scalar
	/// \param right RHS complex number
	/// \return Quotient of LHS and RHS
	template<typename T, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/(const R &left,
															 const Complex<T> &right) {
		Complex<T> tmp(left);
		tmp /= right;
		return tmp;
	}

	/// \brief Equality comparison of two complex numbers
	/// \tparam L Scalar type of LHS complex number
	/// \tparam R Scalar type of RHS complex number
	/// \param left LHS complex number
	/// \param right RHS complex number
	/// \return true if equal, false otherwise
	template<typename L, typename R>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr bool operator==(const Complex<L> &left,
																		const Complex<R> &right) {
		return left.real() == right.real() && left.imag() == right.imag();
	}

	/// \brief Equality comparison of complex number and scalar
	///
	/// Compares the real component of the complex number to the scalar, and the imaginary component
	/// to zero. Returns true if and only if both comparisons are true.
	///
	/// \tparam T Scalar type of complex number
	/// \param left LHS complex number
	/// \param right RHS scalar
	/// \return true if equal, false otherwise
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr bool operator==(const Complex<T> &left,
																		T &right) {
		return left.real() == right && left.imag() == 0;
	}

#if !defined(LIBRAPID_CXX_20)

	/// \brief Equality comparison of scalar and complex number
	///
	/// Compares the real component of the complex number to the scalar, and the imaginary component
	/// to zero. Returns true if and only if both comparisons are true.
	///
	/// \tparam T Scalar type of complex number
	/// \param left LHS scalar
	/// \param right RHS complex number
	/// \return true if equal, false otherwise
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr bool operator==(const T &left,
																		const Complex<T> &right) {
		return left == right.real() && 0 == right.imag();
	}
#endif

#if !defined(LIBRAPID_CXX_20)

	/// \brief Inequality comparison of two complex numbers
	/// \tparam T Scalar type of complex number
	/// \param left LHS complex number
	/// \param right RHS complex number
	/// \return true if ***not*** equal, false otherwise
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr bool operator!=(const Complex<T> &left,
																		const Complex<T> &right) {
		return !(left == right);
	}

	/// \brief Inequality comparison of complex number and scalar
	/// \see operator==(const Complex<T> &, T &)
	/// \tparam T Scalar type of complex number
	/// \param left LHS complex number
	/// \param right RHS scalar
	/// \return true if ***not*** equal, false otherwise
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr bool operator!=(const Complex<T> &left,
																		T &right) {
		return !(left == right);
	}

	/// \brief Inequality comparison of scalar and complex number
	/// \see operator==(const T &, const Complex<T> &)
	/// \tparam T Scalar type of complex number
	/// \param left LHS scalar
	/// \param right RHS complex number
	/// \return true if ***not*** equal, false otherwise
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr bool operator!=(const T &left,
																		const Complex<T> &right) {
		return !(left == right);
	}
#endif

	/// \brief Return \f$ \mathrm{Re}(z) \f$
	/// \tparam T Scalar type of the complex number
	/// \param val Complex number
	/// \return Real component of the complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T real(const Complex<T> &val) {
		return val.real();
	}

	/// \brief Return \f$ \mathrm{Im}(z) \f$
	/// \tparam T Scalar type of the complex number
	/// \param val Complex number
	/// \return Imaginary component of the complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T imag(const Complex<T> &val) {
		return val.imag();
	}

	/// \brief Return \f$ \sqrt{z} \f$
	/// \tparam T Scalar type of the complex number
	/// \param val Complex number
	/// \return Square root of the complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T>
	sqrt(const Complex<T> &val); // Defined later

	/// \brief Return \f$ \sqrt{\mathrm{Re}(z)^2 + \mathrm{Im}(z)^2} \f$
	/// \tparam T Scalar type of the complex number
	/// \param val Complex number
	/// \return Absolute value of the complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T abs(const Complex<T> &val) {
		return ::librapid::hypot(val.real(), val.imag());
	}

	/// \brief Returns \f$z^{*}\f$
	/// \tparam T Scalar type of the complex number
	/// \param val Complex number
	/// \return Complex conjugate of the complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> conj(const Complex<T> &val) {
		return Complex<T>(val.real(), -val.imag());
	}

	/// \brief Compute the complex arc cosine of a complex number
	///
	/// This function computes the complex arc cosine of the input complex number,
	/// \f$z = \text{acos}(z)\f$
	///
	/// The algorithm handles NaN and infinity values, and avoids overflow.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex arc cosine of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> acos(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(typetraits::TypeInfo<T>::max());
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

		if (::librapid::isNaN(re) || ::librapid::isNaN(im)) { // At least one NaN
			ux = typetraits::TypeInfo<T>::quietNaN();
			vx = ux;
		} else if (::librapid::isInf(re)) { // +/- Inf
			if (::librapid::isInf(im)) {
				if (re < 0)
					ux = T(0.75) * pi; // (-Inf, +/-Inf)
				else
					ux = T(0.25) * pi; // (-Inf, +/-Inf)
			} else if (re < 0) {
				ux = pi;			   // (-Inf, finite)
			} else {
				ux = 0;				   // (+Inf, finite)
			}
			vx = -::librapid::copySign(typetraits::TypeInfo<T>::infinity(), im);
		} else if (::librapid::isInf(im)) { // finite, Inf)
			ux = T(0.5) * pi;				// (finite, +/-Inf)
			vx = -im;
		} else {							// (finite, finite)
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
			} else {					   // Shouldn't overflow (?)
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

	/// \brief Compute the complex hyperbolic arc cosine of a complex number
	///
	///
	/// This function computes the complex area hyperbolic cosine of the input complex number,
	/// \f$ z = \text{acosh}(z) \f$
	///
	/// The algorithm handles NaN and infinity values, and avoids overflow.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex area hyperbolic cosine of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> acosh(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(typetraits::TypeInfo<T>::max());
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

		if (::librapid::isNaN(re) || ::librapid::isNaN(im)) { // At least one NaN
			ux = typetraits::TypeInfo<T>::quietNaN();
			vx = ux;
		} else if (::librapid::isInf(re)) { // (+/-Inf, not NaN)
			ux = typetraits::TypeInfo<T>::infinity();
			if (::librapid::isInf(im)) {
				if (re < 0)
					vx = T(0.75) * pi; // (-Inf, +/-Inf)
				else
					vx = T(0.25) * pi; // (+Inf, +/-Inf)
			} else if (re < 0) {
				vx = pi;			   // (-Inf, finite)
			} else {
				vx = 0;				   // (+Inf, finite)
			}
			vx = ::librapid::copySign(vx, im);
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
			} else {					   // Shouldn't overflow (?)
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

	/// \brief Compute the complex arc hyperbolic sine of a complex number
	///
	/// This function computes the complex arc hyperbolic sine of the input complex number,
	/// \f$ z = \text{asinh}(z) \f$
	///
	/// The algorithm handles NaN and infinity values, and avoids overflow.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex arc hyperbolic sine of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> asinh(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(typetraits::TypeInfo<T>::max());
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

		if (::librapid::isNaN(re) || ::librapid::isNaN(im)) { // At least one NaN/Inf
			ux = typetraits::TypeInfo<T>::quietNaN();
			vx = ux;
		} else if (::librapid::isInf(re)) { // (+/-Inf, not NaN)
			if (::librapid::isInf(im)) {	// (+/-Inf, +/-Inf)
				ux = re;
				vx = ::librapid::copySign(T(0.25) * pi, im);
			} else { // (+/-Inf, finite)
				ux = re;
				vx = ::librapid::copySign(T(0), im);
			}
		} else if (::librapid::isInf(im)) {
			ux = ::librapid::copySign(typetraits::TypeInfo<T>::infinity(), re);
			vx = ::librapid::copySign(T(0.5) * pi, im);
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
			} else {					   // Shouldn't overflow (?)
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

	/// \brief Compute the complex arc sine of a complex number
	///
	/// This function computes the complex arc sine of the input complex number,
	/// \f$ z = \text{asin}(z) \f$
	///
	/// It calculates the complex arc sine by using the complex hyperbolic sine function.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex arc sine of the input complex number
	/// \see asinh
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> asin(const Complex<T> &other) {
		Complex<T> asinhVal = asinh(Complex<T>(-imag(other), real(other)));
		return Complex<T>(imag(asinhVal), -real(asinhVal));
	}

	/// \brief Compute the complex arc hyperbolic tangent of a complex number
	///
	/// This function computes the complex arc hyperbolic tangent of the input complex number,
	/// \f$ z = \text{atanh}(z) \f$
	///
	/// This function performs error checking and supports NaNs and Infs.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex arc hyperbolic tangent of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> atanh(const Complex<T> &other) {
		const T arcBig = T(0.25) * ::librapid::sqrt(typetraits::TypeInfo<T>::max());
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

		if (::librapid::isNaN(re) || ::librapid::isNaN(im)) { // At least one NaN
			ux = typetraits::TypeInfo<T>::quietNaN();
			vx = ux;
		} else if (::librapid::isInf(re)) { // (+/-Inf, not NaN)
			ux = ::librapid::copySign(T(0), re);
			vx = ::librapid::copySign(piBy2, im);
		} else { // (finite, not NaN)
			const T magIm = ::librapid::abs(im);
			const T oldRe = re;

			re = ::librapid::abs(re);

			if (arcBig < re) { // |re| is large
				T fx = im / re;
				ux	 = 1 / re / (1 + fx * fx);
				vx	 = ::librapid::copySign(piBy2, im);
			} else if (arcBig < magIm) { // |im| is large
				T fx = re / im;
				ux	 = fx / im / (1 + fx * fx);
				vx	 = ::librapid::copySign(piBy2, im);
			} else if (re != 1) { // |re| is small
				T reFrom1 = 1 - re;
				T imEps2  = magIm * magIm;
				ux = T(0.25) * detail::algorithm::logP1(4 * re / (reFrom1 * reFrom1 + imEps2));
				vx = T(0.5) * ::librapid::atan2(2 * im, reFrom1 * (1 + re) - imEps2);
			} else if (im == 0) { // {+/-1, 0)
				ux = typetraits::TypeInfo<T>::infinity();
				vx = im;
			} else { // (+/-1, nonzero)
				ux = ::librapid::log(::librapid::sqrt(::librapid::sqrt(4 + im * im)) /
									 ::librapid::sqrt(magIm));
				vx = ::librapid::copySign(T(0.5) * (piBy2 + ::librapid::atan2(magIm, T(2))), im);
			}
			ux = ::librapid::copySign(ux, oldRe);
		}
		return Complex<T>(ux, vx);
	}

	/// \brief Compute the complex arc tangent of a complex number
	///
	/// This function computes the complex arc tangent of the input complex number,
	/// \f$ z = \text{atan}(z) \f$
	///
	/// The algorithm handles NaN and infinity values, and avoids overflow.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex arc tangent of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> atan(const Complex<T> &other) {
		Complex atanhVal = ::librapid::atanh(Complex<T>(-imag(other), real(other)));
		return Complex<T>(imag(atanhVal), -real(atanhVal));
	}

	/// \brief Compute the complex hyperbolic cosine of a complex number
	///
	/// This function computes the complex hyperbolic cosine of the input complex number,
	/// \f$ z = \text{cosh}(z) \f$
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex hyperbolic cosine of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> cosh(const Complex<T> &other) {
		return Complex<T>(::librapid::cosh(real(other)) * ::librapid::cos(imag(other)),
						  ::librapid::sinh(real(other)) * ::librapid::sin(imag(other)));
	}

	template<typename T>
	LIBRAPID_NODISCARD Complex<T> polarPositiveNanInfZeroRho(const T &rho, const T &theta) {
		// Rho is +NaN/+Inf/+0
		if (::librapid::isNaN(theta) || ::librapid::isInf(theta)) {		  // Theta is NaN/Inf
			if (::librapid::isInf(rho)) {
				return Complex<T>(rho, ::librapid::sin(theta));			  // (Inf, NaN/Inf)
			} else {
				return Complex<T>(rho, ::librapid::copySign(rho, theta)); // (NaN/0, NaN/Inf)
			}
		} else if (theta == T(0)) {										  // Theta is zero
			return Complex<T>(rho, theta);								  // (NaN/Inf/0, 0)
		} else { // Theta is finite non-zero
			// (NaN/Inf/0, finite non-zero)
			return Complex<T>(rho * ::librapid::cos(theta), rho * ::librapid::sin(theta));
		}
	}

	/// \brief Compute the complex exponential of a complex number
	///
	/// This function computes the complex exponential of the input complex number,
	/// \f$ z = e^z \f$
	///
	/// The algorithm handles NaN and infinity values.
	///
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex exponential of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> exp(const Complex<T> &other) {
		const T logRho = real(other);
		const T theta  = imag(other);

		if (!::librapid::isNaN(logRho) && !::librapid::isInf(logRho)) { // Real component is finite
			T real = logRho;
			T imag = logRho;
			detail::algorithm::expMul(&real, static_cast<T>(::librapid::cos(theta)), 0);
			detail::algorithm::expMul(&imag, static_cast<T>(::librapid::sin(theta)), 0);
			return Complex<T>(real, imag);
		}

		// Real component is NaN/Inf
		// Return polar(exp(re), im)
		if (::librapid::isInf(logRho)) {
			if (logRho < 0) {
				return polarPositiveNanInfZeroRho(T(0), theta);	  // exp(-Inf) = +0
			} else {
				return polarPositiveNanInfZeroRho(logRho, theta); // exp(+Inf) = +Inf
			}
		} else {
			return polarPositiveNanInfZeroRho(static_cast<T>(::librapid::abs(logRho)),
											  theta); // exp(NaN) = +NaN
		}
	}

	/// \brief Compute the complex exponential base 2 of a complex number
	/// \see exp
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex exponential base 2 of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> exp2(const Complex<T> &other) {
		return pow(T(2), other);
	}

	/// \brief Compute the complex exponential base 10 of a complex number
	/// \see exp
	/// \tparam T Scalar type of the complex number
	/// \param other Input complex number
	/// \return Complex exponential base 10 of the input complex number
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> exp10(const Complex<T> &other) {
		return pow(T(10), other);
	}

	template<typename T>
	T _fabs(const Complex<T> &other, int64_t *exp) {
		*exp = 0;
		T av = ::librapid::abs(real(other));
		T bv = ::librapid::abs(imag(other));

		if (::librapid::isInf(av) || ::librapid::isInf(bv)) {
			return typetraits::TypeInfo<T>::infinity(); // At least one component is Inf
		} else if (::librapid::isNaN(av)) {
			return av;									// Real component is NaN
		} else if (::librapid::isNaN(bv)) {
			return bv;									// Imaginary component is NaN
		} else {
			if (av < bv) std::swap(av, bv);
			if (av == 0) return av; // |0| = 0

			if (1 <= av) {
				*exp = 4;
				av	 = av * T(0.0625);
				bv	 = bv * T(0.0625);
			} else {
				const T fltEps	= typetraits::TypeInfo<T>::epsilon();
				const T legTiny = fltEps == 0 ? T(0) : 2 * typetraits::TypeInfo<T>::min() / fltEps;

				if (av < legTiny) {
					int64_t exponent;
#if defined(LIBRAPID_USE_MULTIPREC)
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
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T _logAbs(const Complex<T> &other) noexcept {
		return static_cast<T>(detail::algorithm::logHypot(static_cast<double>(real(other)),
														  static_cast<double>(imag(other))));
	}

#if defined(LIBRAPID_USE_MULTIPREC)
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE mpfr _logAbs(const Complex<mpfr> &other) noexcept {
		return detail::algorithm::logHypot(real(other), imag(other));
	}
#endif

	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE float _logAbs(const Complex<float> &other) noexcept {
		return detail::algorithm::logHypot(real(other), imag(other));
	}

	/// \brief Calculates the natural logarithm of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return Natural logarithm of the complex number
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> log(const Complex<T> &other) {
		const T logAbs = _logAbs(other);
		const T theta  = ::librapid::atan2(imag(other), real(other));
		return Complex<T>(logAbs, theta);
	}

	/// \brief Calculates the logarithm of a complex number with a complex base
	///
	/// \f$ \log_{\mathrm{base}}(z) = \log(z) / \log(\mathrm{base}) \f$
	/// \tparam T Scalar type
	/// \tparam B Base type
	/// \param other Complex number
	/// \param base Base of the logarithm
	/// \return Logarithm of the complex number with the given base
	/// \see log
	template<typename T, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> log(const Complex<T> &other,
															 const Complex<T> &base) {
		return log(other) / log(base);
	}

	/// \brief Calculates the logarithm of a complex number with a real base
	///
	/// \f$ \log_{\mathrm{base}}(z) = \log(z) / \log(\mathrm{base}) \f$
	/// \tparam T Scalar type of the complex number
	/// \tparam B Scalar type of the base
	/// \param other Complex number
	/// \param base Base of the logarithm (real)
	/// \return Logarithm of the complex number with the given base
	/// \see log
	template<typename T, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> log(const Complex<T> &other,
															 const B &base) {
		const T logAbs = _logAbs(other);
		const T theta  = ::librapid::atan2(imag(other), real(other));
		return Complex<T>(logAbs, theta) / ::librapid::log(base);
	}

	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> _pow(const T &left, const T &right) {
		if (0 <= left) {
			return Complex<T>(::librapid::pow(left, right), ::librapid::copySign(T(0), right));
		} else {
			return exp(right * log(Complex<T>(left)));
		}
	}

	/// \brief Calculate \f$ \text{left}^{\text{right}} \f$ for a complex-valued left-hand side
	/// \tparam T Value type for the left-hand side
	/// \tparam V Value type for the right-hand side
	/// \param left Complex base
	/// \param right Real exponent
	/// \return \f$ \text{left}^{\text{right}} \f$
	template<typename T, typename V,
			 typename std::enable_if_t<
			   typetraits::TypeInfo<V>::type == detail::LibRapidType::Scalar, int> = 0>
	LIBRAPID_NODISCARD Complex<T> pow(const Complex<T> &left, const V &right) {
		if (imag(left) == 0) {
			if (::librapid::signBit(imag(left))) {
				return conj(_pow(real(left), static_cast<T>(right)));
			} else {
				return _pow(real(left), static_cast<T>(right));
			}
		} else {
			return exp(static_cast<T>(right) * log(left));
		}
	}

	/// \brief Calculate \f$ \text{left}^{\text{right}} \f$ for a complex-valued right-hand side
	/// \tparam T Value type for the left-hand side
	/// \tparam V Value type for the right-hand side
	/// \param left Real base
	/// \param right Complex exponent
	/// \return \f$ \text{left}^{\text{right}} \f$
	template<typename T, typename V,
			 typename std::enable_if_t<
			   typetraits::TypeInfo<V>::type == detail::LibRapidType::Scalar, int> = 0>
	LIBRAPID_NODISCARD Complex<T> pow(const V &left, const Complex<T> &right) {
		if (imag(right) == 0) {
			return _pow(static_cast<V>(left), real(right));
		} else if (0 < left) {
			return exp(right * ::librapid::log(static_cast<V>(left)));
		} else {
			return exp(right * log(Complex<T>(static_cast<V>(left))));
		}
	}

	/// \brief Calculate \f$ \text{left}^{\text{right}} \f$ for complex numbers
	/// \tparam T Complex number component type
	/// \param left Complex base
	/// \param right Complex exponent
	/// \return \f$ \text{left}^{\text{right}} \f$
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> pow(const Complex<T> &left, const Complex<T> &right) {
		if (imag(right) == 0) {
			return pow(left, real(right));
		} else if (imag(left) == 0 && 0 < real(left)) {
			return exp(right * ::librapid::log(real(left)));
		} else {
			return exp(right * log(left));
		}
	}

	/// \brief Calculate the hyperbolic sine of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \sinh(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> sinh(const Complex<T> &other) {
		return Complex<T>(::librapid::sinh(real(other)) * ::librapid::cos(imag(other)),
						  ::librapid::cosh(real(other)) * ::librapid::sin(imag(other)));
	}

	template<typename T>
	LIBRAPID_NODISCARD Complex<T> sqrt(const Complex<T> &other) {
		int64_t otherExp;
		T rho = _fabs(other, &otherExp); // Get magnitude and scale factor

		if (otherExp == 0) {			 // Argument is zero, Inf or NaN
			if (rho == 0) {
				return Complex<T>(T(0), imag(other));
			} else if (::librapid::isInf(rho)) {
				const T re = real(other);
				const T im = imag(other);

				if (::librapid::isInf(im)) {
					return Complex<T>(typetraits::TypeInfo<T>::infinity(), im); // (any, +/-Inf)
				} else if (::librapid::isNaN(im)) {
					if (re < 0) {
						// (-Inf, NaN)
						return Complex<T>(::librapid::abs(im), ::librapid::copySign(re, im));
					} else {
						return other; // (+Inf, NaN)
					}
				} else {
					if (re < 0) {
						return Complex<T>(T(0), ::librapid::copySign(re, im)); // (-Inf, finite)
					} else {
						return Complex<T>(re, ::librapid::copySign(T(0), im)); // (+Inf, finite)
					}
				}
			} else {
				return Complex<T>(rho, rho);
			}
		} else { // Compute in safest quadrant
			T realMag = ::librapid::ldexp(::librapid::abs(real(other)), -otherExp);
			rho		  = ::librapid::ldexp(::librapid::sqrt(2 * (realMag + rho)), otherExp / 2 - 1);
			if (0 <= real(other)) {
				return Complex<T>(rho, imag(other) / (2 * rho));
			} else {
				return Complex<T>(::librapid::abs(imag(other) / (2 * rho)),
								  ::librapid::copySign(rho, imag(other)));
			}
		}
	}

	/// \brief Calculate the hyperbolic tangent of a complex number
	///
	/// This function supports propagation of NaNs and Infs.
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \tanh(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> tanh(const Complex<T> &other) {
		T tv = ::librapid::tan(imag(other));
		T sv = ::librapid::sinh(real(other));
		T bv = sv * (T(1) + tv * tv);
		T dv = T(1) + bv * sv;

		if (::librapid::isInf(dv)) {
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

	/// \brief Return the phase angle of a complex value as a real
	///
	/// This function calls \f$ \text{atan2}(\text{imag}(z), \text{real}(z)) \f$.
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \arg(z) \f$
	/// \see atan2
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T arg(const Complex<T> &other) {
		return ::librapid::atan2(imag(other), real(other));
	}

	/// \brief Project a complex number onto the Riemann sphere
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \text{proj}(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> proj(const Complex<T> &other) {
		if (::librapid::isInf(real(other)) || ::librapid::isInf(imag(other))) {
			const T im = ::librapid::copySign(T(0), imag(other));
			return Complex<T>(typetraits::TypeInfo<T>::infinity(), im);
		}
		return other;
	}

	/// \brief Calculate the cosine of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \cos(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> cos(const Complex<T> &other) {
		return Complex<T>(::librapid::cosh(imag(other)) * ::librapid::cos(real(other)),
						  -::librapid::sinh(imag(other)) * ::librapid::sin(real(other)));
	}

	/// \brief Calculate the cosecant of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \csc(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> csc(const Complex<T> &other) {
		return T(1) / sin(other);
	}

	/// \brief Calculate the secant of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \sec(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> sec(const Complex<T> &other) {
		return T(1) / cos(other);
	}

	/// \brief Calculate the cotangent of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \cot(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> cot(const Complex<T> &other) {
		return T(1) / tan(other);
	}

	/// \brief Calculate the arc cosecant of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \operatorname{arccsc}(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> acsc(const Complex<T> &other) {
		return asin(T(1) / other);
	}

	/// \brief Calculate the arc secant of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \operatorname{arcsec}(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> asec(const Complex<T> &other) {
		return acos(T(1) / other);
	}

	/// \brief Calculate the arc cotangent of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \operatorname{arccot}(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> acot(const Complex<T> &other) {
		return atan(T(1) / other);
	}

	/// \brief Calculate the logarithm base 2 of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \log_2(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> log2(const Complex<T> &other) {
		return log(other) / ::librapid::log(T(2));
	}

	/// \brief Calculate the logarithm base 10 of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \log_{10}(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> log10(const Complex<T> &other) {
		return log(other) / ::librapid::log(10);
	}

	// Return magnitude squared

	/// \brief Calculate the magnitude squared of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ |z|^2 \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T norm(const Complex<T> &other) {
		return real(other) * real(other) + imag(other) * imag(other);
	}

	/// \brief Return a complex number from polar coordinates
	///
	/// Given a radius, \p rho, and an angle, \p theta, this function returns the complex number
	/// \f$ \rho e^{i\theta} \f$.
	///
	/// The function returns NaN, infinity or zero based on the input values of rho.
	/// \tparam T Scalar type of the complex number
	/// \param rho Radius of the polar coordinate system
	/// \param theta Angle of the polar coordinate system
	/// \return Complex number in polar form.
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> polar(const T &rho, const T &theta) {
		if (!::librapid::isNaN(rho) && !::librapid::isInf(rho) && rho != T(0)) {
			// Rho is finite and non-zero
			return Complex<T>(rho * ::librapid::cos(theta), rho * ::librapid::sin(theta));
		}

		// Rho is NaN/Inf/0
		if (::librapid::signBit(rho))
			return -polarPositiveNanInfZeroRho(-rho, theta);
		else
			return polarPositiveNanInfZeroRho(rho, theta);
	}

	/// \brief Compute the sine of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \sin(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> sin(const Complex<T> &other) {
		return Complex<T>(::librapid::cosh(imag(other)) * ::librapid::sin(real(other)),
						  ::librapid::sinh(imag(other)) * ::librapid::cos(real(other)));
	}

	/// \brief Compute the tangent of a complex number
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ \tan(z) \f$
	template<typename T>
	LIBRAPID_NODISCARD Complex<T> tan(const Complex<T> &other) {
		Complex<T> zv(tanh(Complex<T>(-imag(other), real(other))));
		return Complex<T>(imag(zv), -real(zv));
	}

	/// \brief Round the real and imaginary parts of a complex number towards \f$ -\infty \f$
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$ (\lfloor\operatorname{real}(z)\rfloor,\lfloor\operatorname{imag}(z)\rfloor )\f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> floor(const Complex<T> &other) {
		return Complex<T>(::librapid::floor(real(other)), ::librapid::floor(imag(other)));
	}

	/// \brief Round the real and imaginary parts of a complex number towards \f$ +\infty \f$
	/// \tparam T Scalar type
	/// \param other Complex number
	/// \return \f$(\lceil\operatorname{real}(z)\rceil,\lceil\operatorname{imag}(z)\rceil )\f$
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> ceil(const Complex<T> &other) {
		return Complex<T>(::librapid::ceil(real(other)), ::librapid::ceil(imag(other)));
	}

	/// \brief Generate a random complex number between two given complex numbers
	///
	/// This function generates a random complex number in the range [min, max], where min
	/// and max are given as input. The function uses a default seed if none is provided.
	///
	/// \tparam T Scalar type of the complex number
	/// \param min Minimum complex number
	/// \param max Maximum complex number
	/// \param seed Seed for the random number generator
	/// \return Random complex number between min and max
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T>
	random(const Complex<T> &min, const Complex<T> &max, uint64_t seed = -1) {
		return Complex<T>(::librapid::random(real(min), real(max), seed),
						  ::librapid::random(imag(min), imag(max), seed));
	}

	namespace typetraits {
		template<typename T>
		struct TypeInfo<Complex<T>> {
			detail::LibRapidType type = detail::LibRapidType::Scalar;
			using Scalar			  = Complex<T>;
			using Packet =
			  typename std::conditional_t<(TypeInfo<T>::packetWidth > 1),
										  Complex<typename TypeInfo<T>::Packet>, std::false_type>;
			static constexpr int64_t packetWidth =
			  TypeInfo<typename TypeInfo<T>::Scalar>::packetWidth;
			static constexpr char name[]			 = "Complex";
			static constexpr bool supportsArithmetic = true;
			static constexpr bool supportsLogical	 = true;
			static constexpr bool supportsBinary	 = false;
			static constexpr bool allowVectorisation = false;

#if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_C_64F;
#endif

			static constexpr bool canAlign	= TypeInfo<T>::canAlign;
			static constexpr bool canMemcpy = TypeInfo<T>::canMemcpy;

			LIMIT_IMPL(min) { return TypeInfo<T>::min(); }
			LIMIT_IMPL(max) { return TypeInfo<T>::max(); }
			LIMIT_IMPL(epsilon) { return TypeInfo<T>::epsilon(); }
			LIMIT_IMPL(roundError) { return TypeInfo<T>::roundError(); }
			LIMIT_IMPL(denormMin) { return TypeInfo<T>::denormMin(); }
			LIMIT_IMPL(infinity) { return TypeInfo<T>::infinity(); }
			LIMIT_IMPL(quietNaN) { return TypeInfo<T>::quietNaN(); }
			LIMIT_IMPL(signalingNaN) { return TypeInfo<T>::signalingNaN(); }
		};
	} // namespace typetraits
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
LIBRAPID_SIMPLE_IO_IMPL(typename Scalar, librapid::Complex<Scalar>)
#endif // FMT_API

#ifdef USE_X86_X64_INTRINSICS
#	undef USE_X86_X64_INTRINSICS
#endif

#ifdef USE_ARM64_INTRINSICS
#	undef USE_ARM64_INTRINSICS
#endif

#endif // LIBRAPID_MATH_COMPLEX_HPP