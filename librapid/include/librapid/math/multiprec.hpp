#ifndef LIBRAPID_MATH_MULTIPREC_HPP
#define LIBRAPID_MATH_MULTIPREC_HPP

#if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
	/// Multiprecision integer type
	using mpz = mpz_class;
	/// Multiprecision floating point type
	using mpf = mpf_class;
	/// Multiprecision rational type
	using mpq = mpq_class;
	/// Multiprecision floating point type with greater functionality
	using mpfr = mpfr::mpreal;

	/// Convert a multiprecision integer type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpz &val, int64_t digits = -1, int base = 10);

	/// Convert a multiprecision floating point type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpf &val, int64_t digits = -1, int base = 10);

	/// Convert a multiprecision rational type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpq &val, int64_t digits = -1, int base = 10);

	/// Convert a multiprecision floating point type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpfr &val, int64_t digits = -1, int base = 10);

	/// Multiprecision integer to multiprecision integer cast
	/// \param other The value to cast
	/// \return The cast value
	mpz toMpz(const mpz &other);

	/// Multiprecision floating point to multiprecision integer cast
	/// \param other The value to cast
	/// \return The cast value
	mpz toMpz(const mpf &other);

	/// Multiprecision rational to multiprecision integer cast
	/// \param other The value to cast
	/// \return The cast value
	mpz toMpz(const mpq &other);

	/// Multiprecision floating point to multiprecision integer cast
	/// \param other The value to cast
	/// \return The cast value
	mpz toMpz(const mpfr &other);

	/// Multiprecision integer to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpf toMpf(const mpz &other);

	/// Multiprecision floating point to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpf toMpf(const mpf &other);

	/// Multiprecision rational to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpf toMpf(const mpq &other);

	/// Multiprecision floating point to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpf toMpf(const mpfr &other);

	/// Multiprecision integer to multiprecision rational cast
	/// \param other The value to cast
	/// \return The cast value

	mpq toMpq(const mpz &other);

	/// Multiprecision floating point to multiprecision rational cast
	/// \param other The value to cast
	/// \return The cast value
	mpq toMpq(const mpf &other);

	/// Multiprecision rational to multiprecision rational cast
	/// \param other The value to cast
	/// \return The cast value
	mpq toMpq(const mpq &other);

	/// Multiprecision floating point to multiprecision rational cast
	/// \param other The value to cast
	/// \return The cast value
	mpq toMpq(const mpfr &other);

	/// Multiprecision integer to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpfr toMpfr(const mpz &other);

	/// Multiprecision floating point to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpfr toMpfr(const mpf &other);

	/// Multiprecision rational to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpfr toMpfr(const mpq &other);

	/// Multiprecision floating point to multiprecision floating point cast
	/// \param other The value to cast
	/// \return The cast value
	mpfr toMpfr(const mpfr &other);

	// Trigonometric Functionality for mpf

	/// Sine of a multiprecision floating point value: \f$ \sin (x) \f$
	/// \param val The value to take the sine of
	/// \return The sine of the value
	mpfr sin(const mpfr &val);

	/// Cosine of a multiprecision floating point value: \f$ \cos (x) \f$
	/// \param val The value to take the cosine of
	/// \return The cosine of the value
	mpfr cos(const mpfr &val);

	/// Tangent of a multiprecision floating point value: \f$ \tan (x) \f$
	/// \param val The value to take the tangent of
	/// \return The tangent of the value
	mpfr tan(const mpfr &val);

	/// Arcsine of a multiprecision floating point value: \f$ \sin^{-1} (x) \f$
	/// \param val The value to take the arcsine of
	/// \return The arcsine of the value
	/// \see sin(const mpfr &)
	mpfr asin(const mpfr &val);

	/// Arccosine of a multiprecision floating point value: \f$ \cos^{-1} (x) \f$
	/// \param val The value to take the arccosine of
	/// \return The arccosine of the value
	/// \see cos(const mpfr &)
	mpfr acos(const mpfr &val);

	/// Arctangent of a multiprecision floating point value: \f$ \tan^{-1} (x) \f$
	/// \param val The value to take the arctangent of
	/// \return The arctangent of the value
	/// \see tan(const mpfr &)
	mpfr atan(const mpfr &val);

	/// Atan2 of a multiprecision floating point value: \f$ \tan^{-1}\left(\frac{y}{x}\right) \f$
	/// \param dy The y value
	/// \param dx The x value
	/// \return The atan2 of the value
	mpfr atan2(const mpfr &dy, const mpfr &dx);

	/// Cosec of a multiprecision floating point value: \f$ \csc (x) \f$
	/// \param val The value to take the cosec of
	/// \return The cosec of the value
	mpfr csc(const mpfr &val);

	/// Secant of a multiprecision floating point value: \f$ \sec (x) \f$
	/// \param val The value to take the secant of
	/// \return The secant of the value
	mpfr sec(const mpfr &val);

	/// Cotangent of a multiprecision floating point value: \f$ \cot (x) \f$
	/// \param val The value to take the cotangent of
	/// \return The cotangent of the value
	mpfr cot(const mpfr &val);

	/// Arccosec of a multiprecision floating point value: \f$ \csc^{-1} (x) \f$
	/// \param val The value to take the arccosec of
	/// \return The arccosec of the value
	mpfr acsc(const mpfr &val);

	/// Arcsecant of a multiprecision floating point value: \f$ \sec^{-1} (x) \f$
	/// \param val The value to take the arcsecant of
	/// \return The arcsecant of the value
	mpfr asec(const mpfr &val);

	/// Arccotangent of a multiprecision floating point value: \f$ \cot^{-1} (x) \f$
	/// \param val The value to take the arccotangent of
	/// \return The arccotangent of the value
	mpfr acot(const mpfr &val);

	// Hyperbolic Functionality for mpf

	/// Hyperbolic sine of a multiprecision floating point value: \f$ \sinh (x) \f$
	/// \param val The value to take the hyperbolic sine of
	/// \return The hyperbolic sine of the value
	mpfr sinh(const mpfr &val);

	/// Hyperbolic cosine of a multiprecision floating point value: \f$ \cosh (x) \f$
	/// \param val The value to take the hyperbolic cosine of
	/// \return The hyperbolic cosine of the value
	mpfr cosh(const mpfr &val);

	/// Hyperbolic tangent of a multiprecision floating point value: \f$ \tanh (x) \f$
	/// \param val The value to take the hyperbolic tangent of
	/// \return The hyperbolic tangent of the value
	mpfr tanh(const mpfr &val);

	/// Hyperbolic arcsine of a multiprecision floating point value: \f$ \sinh^{-1} (x) \f$
	/// \param val The value to take the hyperbolic arcsine of
	/// \return The hyperbolic arcsine of the value
	mpfr asinh(const mpfr &val);

	/// Hyperbolic arccosine of a multiprecision floating point value: \f$ \cosh^{-1} (x) \f$
	/// \param val The value to take the hyperbolic arccosine of
	/// \return The hyperbolic arccosine of the value
	mpfr acosh(const mpfr &val);

	/// Hyperbolic arctangent of a multiprecision floating point value: \f$ \tanh^{-1} (x) \f$
	/// \param val The value to take the hyperbolic arctangent of
	/// \return The hyperbolic arctangent of the value
	mpfr atanh(const mpfr &val);

	/// Hyperbolic cosec of a multiprecision floating point value: \f$ csch(x) \f$
	/// \param val The value to take the hyperbolic cosec of
	/// \return The hyperbolic cosec of the value
	mpfr csch(const mpfr &val);

	/// Hyperbolic secant of a multiprecision floating point value: \f$ sech(x) \f$
	/// \param val The value to take the hyperbolic secant of
	/// \return The hyperbolic secant of the value
	mpfr sech(const mpfr &val);

	/// Hyperbolic cotangent of a multiprecision floating point value: \f$ coth(x) \f$
	/// \param val The value to take the hyperbolic cotangent of
	/// \return The hyperbolic cotangent of the value
	mpfr coth(const mpfr &val);

	/// Hyperbolic arccosec of a multiprecision floating point value: \f$ csch^{-1}(x) \f$
	/// \param val The value to take the hyperbolic arccosec of
	/// \return The hyperbolic arccosec of the value
	mpfr acsch(const mpfr &val);

	/// Hyperbolic arcsecant of a multiprecision floating point value: \f$ sech^{-1}(x) \f$
	/// \param val The value to take the hyperbolic arcsecant of
	/// \return The hyperbolic arcsecant of the value
	mpfr asech(const mpfr &val);

	/// Hyperbolic arccotangent of a multiprecision floating point value: \f$ coth^{-1}(x)
	/// \f$ \param val The value to take the hyperbolic arccotangent of \return The hyperbolic
	/// \return arccotangent of the value
	mpfr acoth(const mpfr &val);

	/// Absolute value of a multiprecision floating point value: \f$ |x| \f$
	/// \param val The value to take the absolute value of
	/// \return Absolute value
	mpfr abs(const mpfr &val);

	/// Absolute value of a multiprecision integer value: \f$ |x| \f$
	/// \param val The value to take the absolute value of
	/// \return Absolute value
	mpz abs(const mpz &val);

	/// Absolute value of a multiprecision rational value: \f$ |x| \f$
	/// \param val The value to take the absolute value of
	/// \return Absolute value
	mpq abs(const mpq &val);

	/// Absolute value of a multiprecision floating point value: \f$ |x| \f$
	/// \param val The value to take the absolute value of
	/// \return Absolute value
	mpf abs(const mpf &val);

	/// Square root of a multiprecision floating point value: \f$ \sqrt{x} \f$
	/// \param val The value to take the square root of
	/// \return The square root of the value
	mpfr sqrt(const mpfr &val);

	/// Raise a multiprecision floating point value to a power: \f$ x^y \f$
	/// \param base The value to raise to a power
	/// \param pow The power to raise the value to
	mpfr pow(const mpfr &base, const mpfr &pow);

	/// Exponential of a multiprecision floating point value: \f$ e^x \f$
	/// \param val The value to take the exponential of
	/// \return The exponential of the value
	mpfr exp(const mpfr &val);

	/// Raise 2 to the power of a multiprecision floating point value: \f$ 2^x \f$
	/// \param val The value to raise 2 to the power of
	/// \return 2 raised to the power of the value
	mpfr exp2(const mpfr &val);

	/// Raise 10 to the power of a multiprecision floating point value: \f$ 10^x \f$
	/// \param val The value to raise 10 to the power of
	/// \return 10 raised to the power of the value
	mpfr exp10(const mpfr &val);

	/// ldexp of a multiprecision floating point value: \f$ x \times 2^exp \f$
	/// \param val The value to take the ldexp of
	/// \param exponent The exponent to multiply the value by
	/// \return The ldexp of the value
	mpfr ldexp(const mpfr &val, int exponent);

	/// Logarithm of a multiprecision floating point value: \f$ \log (x) \f$
	/// \param val The value to take the logarithm of
	/// \return The logarithm of the value
	mpfr log(const mpfr &val);

	/// Logarithm of a multiprecision floating point value with a given base: \f$ \log_b (x) \f$
	/// \param val The value to take the logarithm of
	/// \param base The base to take the logarithm with
	/// \return The logarithm of the value with the given base
	mpfr log(const mpfr &val, const mpfr &base);

	/// Logarithm of a multiprecision floating point value with base 2: \f$ \log_2 (x) \f$
	/// \param val The value to take the logarithm of
	/// \return The logarithm of the value with base 2
	mpfr log2(const mpfr &val);

	/// Logarithm of a multiprecision floating point value with base 10: \f$ \log_{10} (x) \f$
	/// \param val The value to take the logarithm of
	/// \return The logarithm of the value with base 10
	mpfr log10(const mpfr &val);

	/// Floor of a multiprecision floating point value: \f$ \lfloor x \rfloor \f$
	/// \param val The value to take the floor of
	/// \return The floor of the value
	mpfr floor(const mpfr &val);

	/// Ceiling of a multiprecision floating point value: \f$ \lceil x \rceil \f$
	/// \param val The value to take the ceiling of
	/// \return The ceiling of the value
	mpfr ceil(const mpfr &val);

	/// Floating point modulus of a multiprecision floating point value: \f$ x \bmod y \f$
	/// \param val The value to take the modulus of
	/// \param mod The modulus to take the value by
	/// \return The modulus of the value
	mpfr mod(const mpfr &val, const mpfr &mod);

	/// Hypotenuse of a multiprecision floating point value: \f$ \sqrt{a^2 + b^2} \f$
	/// \param a The first value to take the hypotenuse of
	/// \param b The second value to take the hypotenuse of
	/// \return The hypotenuse of the values
	mpfr hypot(const mpfr &a, const mpfr &b);

	/// Calculate and return \f$ \pi \f$ with LibRapid's current precision
	/// \return \f$ \pi \f$
	/// \see prec
	LIBRAPID_ALWAYS_INLINE mpfr constPi() { return ::mpfr::const_pi(); }

	/// Calculate and return \f$ \gamma \f$ with LibRapid's current precision, where \f$ \gamma \f$
	/// is the Euler-Mascheroni constant
	/// \return \f$ \gamma \f$
	/// \see prec
	LIBRAPID_ALWAYS_INLINE mpfr constEulerMascheroni() { return ::mpfr::const_euler(); }

	/// Calculate and return \f$ \log_e(2) \f$ with LibRapid's current precision
	/// \return \f$ \log_e(2) \f$
	/// \see prec
	LIBRAPID_ALWAYS_INLINE mpfr constLog2() { return ::mpfr::const_log2(); }

	/// Calculate and return Catalan's constant \f$ \gamma \f$ with LibRapid's current precision
	/// \return \f$ \gamma \f$
	/// \see prec
	LIBRAPID_ALWAYS_INLINE mpfr constCatalan() { return ::mpfr::const_catalan(); }

	/// Evaluates to true if the given type is a multiprecision value
	/// \tparam T
	template<typename T>
	struct IsMultiprecision : public std::false_type {};

	template<>
	struct IsMultiprecision<mpz> : public std::true_type {};

	template<>
	struct IsMultiprecision<mpf> : public std::true_type {};

	template<>
	struct IsMultiprecision<mpq> : public std::true_type {};

	template<>
	struct IsMultiprecision<mpfr> : public std::true_type {};

	/// Set the number of base 10 digits to store accurately
	/// \param dig10
	inline void prec(int64_t dig10) {
		int64_t dig2 = ::mpfr::digits2bits((int)dig10);
		mpf_set_default_prec(dig2);
		mpfr::mpreal::set_default_prec((mpfr_prec_t)dig2);
	}

	/// Set the number of bits used to represent each number
	/// \param dig2
	inline void prec2(int64_t dig2) {
		mpf_set_default_prec(dig2);
		mpfr::mpreal::set_default_prec((mpfr_prec_t)dig2);
	}

	/// Returns true if the passed value is not a number (NaN)
	/// Note: MPIR does not support NaN, so chances are it'll have errored already...
	/// \tparam A
	/// \tparam B
	/// \param val The value to check
	/// \return True if the value is NaN
	template<typename A, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isNaN(const __gmp_expr<A, B> &val) noexcept {
		return false;
	}

	/// Returns true if the passed value is not a number (NaN)
	/// \param val The value to check
	/// \return True if the value is NaN
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isNaN(const mpfr &val) noexcept {
		return ::mpfr::isnan(val);
	}

	/// Returns true if the passed value is finite.
	/// Note: MPIR does not support Inf, so we can probably just return true
	/// \tparam A
	/// \tparam B
	/// \param val The value to check
	/// \return True if the value is finite
	template<typename A, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isFinite(const __gmp_expr<A, B> &val) noexcept {
		return true;
	}

	/// Returns true if the passed value is finite.
	/// \param val The value to check
	/// \return True if the value is finite
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isFinite(const mpfr &val) noexcept {
		return ::mpfr::isfinite(val);
	}

	/// Returns true if the passed value is infinite.
	/// Note: MPIR does not support Inf, so we can probably just return false
	/// \tparam A
	/// \tparam B
	/// \param val The value to check
	template<typename A, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isInf(const __gmp_expr<A, B> &val) noexcept {
		return false;
	}

	/// Returns true if the passed value is infinite.
	/// \param val The value to check
	/// \return True if the value is infinite
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isInf(const mpfr &val) noexcept {
		return ::mpfr::isinf(val);
	}

	/// Copy the sign of a value to another value
	/// \param mag The magnitude of the returned value
	/// \param sign The sign of the returned value
	/// \return (<sign> <mag>)
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE mpfr copySign(const mpfr &mag,
															const mpfr &sign) noexcept {
		return ::mpfr::copysign(mag, sign);
	}

	/// Copy the sign of a value to another value
	/// \tparam A
	/// \tparam B
	/// \return (<sign> <mag>)
	template<typename A, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE __gmp_expr<A, B>
	copySign(const __gmp_expr<A, B> &mag, const __gmp_expr<A, B> &sign) noexcept {
		if (sign >= 0 && mag >= 0) return mag;
		if (sign >= 0 && mag < 0) return -mag;
		if (sign < 0 && mag >= 0) return -mag;
		if (sign < 0 && mag < 0) return mag;
		return 0; // Should never get here
	}

	/// Extract the sign bit of a value
	/// \tparam A
	/// \tparam B
	/// \param val The value to extract the sign bit from
	/// \return The sign bit of the value
	template<typename A, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool signBit(const __gmp_expr<A, B> &val) noexcept {
		return val < 0 || val == -0.0; // I have no idea if this works
	}

	/// Extract the sign bit of a value
	/// \param val The value to extract the sign bit from
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool signBit(const mpfr &val) noexcept {
		return ::mpfr::signbit(val);
	}

	/// Multiply a value by 2 raised to the power of an exponent
	/// \return x * 2^exp
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE mpfr ldexp(const mpfr &x,
														 const int64_t exp) noexcept {
		return ::mpfr::ldexp(x, static_cast<mp_exp_t>(exp));
	}

	/// Multiply a value by 2 raised to the power of an exponent
	/// \tparam x The value
	/// \tparam exp The exponent
	/// \return x * 2^exp
	template<typename A, typename B>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE __gmp_expr<A, B> ldexp(const __gmp_expr<A, B> &x,
																	 const int64_t exp) noexcept {
		return x << exp;
	}

	namespace typetraits {
		template<>
		struct TypeInfo<mpz> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Scalar;
			using Scalar							   = mpz;
			using Packet							   = std::false_type;
			using Device							   = device::CPU;
			static constexpr int64_t packetWidth	   = 1;
			static constexpr char name[]			   = "mpz";
			static constexpr bool supportsArithmetic   = true;
			static constexpr bool supportsLogical	   = true;
			static constexpr bool supportsBinary	   = true;
			static constexpr bool allowVectorisation   = false;

#	if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64I;
#	endif

			static constexpr bool canAlign	= false;
			static constexpr bool canMemcpy = false;

			LIMIT_IMPL(min) { return NUM_LIM(min); }
			LIMIT_IMPL(max) { return NUM_LIM(max); }
			LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<mpq> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Scalar;
			using Scalar							   = mpq;
			using Packet							   = std::false_type;
			using Device							   = device::CPU;
			static constexpr int64_t packetWidth	   = 1;
			static constexpr char name[]			   = "mpq";
			static constexpr bool supportsArithmetic   = true;
			static constexpr bool supportsLogical	   = true;
			static constexpr bool supportsBinary	   = false;
			static constexpr bool allowVectorisation   = false;

#	if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

			static constexpr bool canAlign	= false;
			static constexpr bool canMemcpy = false;

			LIMIT_IMPL(min) { return NUM_LIM(min); }
			LIMIT_IMPL(max) { return NUM_LIM(max); }
			LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<mpf> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Scalar;
			using Scalar							   = mpf;
			using Packet							   = std::false_type;
			using Device							   = device::CPU;
			static constexpr int64_t packetWidth	   = 1;
			static constexpr char name[]			   = "mpf";
			static constexpr bool supportsArithmetic   = true;
			static constexpr bool supportsLogical	   = true;
			static constexpr bool supportsBinary	   = false;
			static constexpr bool allowVectorisation   = false;

#	if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

			static constexpr bool canAlign	= false;
			static constexpr bool canMemcpy = false;

			LIMIT_IMPL(min) { return NUM_LIM(min); }
			LIMIT_IMPL(max) { return NUM_LIM(max); }
			LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<mpfr> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Scalar;
			using Scalar							   = mpfr;
			using Packet							   = std::false_type;
			using Device							   = device::CPU;
			static constexpr int64_t packetWidth	   = 1;
			static constexpr char name[]			   = "mpfr";
			static constexpr bool supportsArithmetic   = true;
			static constexpr bool supportsLogical	   = true;
			static constexpr bool supportsBinary	   = false;
			static constexpr bool allowVectorisation   = false;

#	if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

			static constexpr bool canAlign	= false;
			static constexpr bool canMemcpy = false;

			LIMIT_IMPL(min) { return NUM_LIM(min); }
			LIMIT_IMPL(max) { return NUM_LIM(max); }
			LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};
	} // namespace typetraits
} // namespace librapid

// Provide {fmt} printing capabilities
#	ifdef FMT_API
template<>
struct fmt::formatter<mpz_class> {
	detail::dynamic_format_specs<char> specs_;

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		auto begin = ctx.begin(), end = ctx.end();
		if (begin == end) return begin;
		using handler_type = detail::dynamic_specs_handler<ParseContext>;
		auto type		   = detail::type_constant<mpz_class, char>::value;
		auto checker	   = detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
		auto it			   = detail::parse_format_specs(begin, end, checker);
		auto eh			   = ctx.error_handler();
		detail::parse_float_type_spec(specs_, eh);
		return it;
	}

	template<typename FormatContext>
	inline auto format(const mpz_class &num, FormatContext &ctx) {
		try {
			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision < 0 ? 10 : specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};

template<>
struct fmt::formatter<mpf_class> {
	detail::dynamic_format_specs<char> specs_;

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		auto begin = ctx.begin(), end = ctx.end();
		if (begin == end) return begin;
		using handler_type = detail::dynamic_specs_handler<ParseContext>;
		auto type		   = detail::type_constant<mpf_class, char>::value;
		auto checker	   = detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
		auto it			   = detail::parse_format_specs(begin, end, checker);
		auto eh			   = ctx.error_handler();
		detail::parse_float_type_spec(specs_, eh);
		return it;
	}

	template<typename FormatContext>
	inline auto format(const mpf_class &num, FormatContext &ctx) {
		try {
			if (specs_.precision < 1) return fmt::format_to(ctx.out(), librapid::str(num));

			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};

template<typename Type, typename Expression>
struct fmt::formatter<__gmp_expr<Type, Expression>> {
	detail::dynamic_format_specs<char> specs_;

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		auto begin = ctx.begin(), end = ctx.end();
		if (begin == end) return begin;
		using handler_type = detail::dynamic_specs_handler<ParseContext>;
		auto type		   = detail::type_constant<mpf_class, char>::value;
		auto checker	   = detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
		auto it			   = detail::parse_format_specs(begin, end, checker);
		auto eh			   = ctx.error_handler();
		detail::parse_float_type_spec(specs_, eh);
		return it;
	}

	template<typename FormatContext>
	inline auto format(const __gmp_expr<Type, Expression> &num, FormatContext &ctx) {
		try {
			if (specs_.precision < 1) return fmt::format_to(ctx.out(), librapid::str(num));

			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};

template<>
struct fmt::formatter<mpq_class> {
	detail::dynamic_format_specs<char> specs_;

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		auto begin = ctx.begin(), end = ctx.end();
		if (begin == end) return begin;
		using handler_type = detail::dynamic_specs_handler<ParseContext>;
		auto type		   = detail::type_constant<mpq_class, char>::value;
		auto checker	   = detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
		auto it			   = detail::parse_format_specs(begin, end, checker);
		auto eh			   = ctx.error_handler();
		detail::parse_float_type_spec(specs_, eh);
		return it;
	}

	template<typename FormatContext>
	inline auto format(const mpq_class &num, FormatContext &ctx) {
		try {
			if (specs_.precision < 1) return fmt::format_to(ctx.out(), librapid::str(num));

			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};

template<>
struct fmt::formatter<librapid::mpfr> {
	detail::dynamic_format_specs<char> specs_;

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		auto begin = ctx.begin(), end = ctx.end();
		if (begin == end) return begin;
		using handler_type = detail::dynamic_specs_handler<ParseContext>;
		auto type		   = detail::type_constant<mpq_class, char>::value;
		auto checker	   = detail::specs_checker<handler_type>(handler_type(specs_, ctx), type);
		auto it			   = detail::parse_format_specs(begin, end, checker);
		auto eh			   = ctx.error_handler();
		detail::parse_float_type_spec(specs_, eh);
		return it;
	}

	template<typename FormatContext>
	inline auto format(const librapid::mpfr &num, FormatContext &ctx) {
		try {
			if (specs_.precision < 1) return fmt::format_to(ctx.out(), librapid::str(num));

			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};
#	endif // FMT_API

#	if defined(SCN_SCN_H)

namespace scn {
	SCN_BEGIN_NAMESPACE

	template<>
	struct scanner<librapid::mpz> : public detail::string_scanner {
		template<typename Context>
		error scan(librapid::mpz &val, Context &ctx) {
			if (set_parser.enabled()) {
				bool loc = (common_options & localized) != 0;
				bool mb =
				  (loc || set_parser.get_option(detail::set_parser_type::flag::use_ranges)) &&
				  detail::is_multichar_type(typename Context::char_type {});
				std::string tmp;
				auto ret = do_scan(ctx, tmp, pred<Context> {ctx, set_parser, loc, mb});
				val		 = librapid::mpz(tmp);
				return ret;
			}

			auto e = skip_range_whitespace(ctx, false);
			if (!e) { return e; }

			auto is_space_pred = detail::make_is_space_predicate(
			  ctx.locale(), (common_options & localized) != 0, field_width);
			std::string tmp;
			auto ret = do_scan(ctx, tmp, is_space_pred);
			val		 = librapid::mpz(tmp);
			return ret;
		}
	};

	template<>
	struct scanner<librapid::mpf> : public detail::string_scanner {
		template<typename Context>
		error scan(librapid::mpf &val, Context &ctx) {
			if (set_parser.enabled()) {
				bool loc = (common_options & localized) != 0;
				bool mb =
				  (loc || set_parser.get_option(detail::set_parser_type::flag::use_ranges)) &&
				  detail::is_multichar_type(typename Context::char_type {});
				std::string tmp;
				auto ret = do_scan(ctx, tmp, pred<Context> {ctx, set_parser, loc, mb});
				val		 = librapid::mpf(tmp);
				return ret;
			}

			auto e = skip_range_whitespace(ctx, false);
			if (!e) { return e; }

			auto is_space_pred = detail::make_is_space_predicate(
			  ctx.locale(), (common_options & localized) != 0, field_width);
			std::string tmp;
			auto ret = do_scan(ctx, tmp, is_space_pred);
			val		 = librapid::mpf(tmp);
			return ret;
		}
	};

	template<>
	struct scanner<librapid::mpq> : public detail::string_scanner {
		template<typename Context>
		error scan(librapid::mpq &val, Context &ctx) {
			if (set_parser.enabled()) {
				bool loc = (common_options & localized) != 0;
				bool mb =
				  (loc || set_parser.get_option(detail::set_parser_type::flag::use_ranges)) &&
				  detail::is_multichar_type(typename Context::char_type {});
				std::string tmp;
				auto ret = do_scan(ctx, tmp, pred<Context> {ctx, set_parser, loc, mb});
				val		 = librapid::mpq(tmp);
				return ret;
			}

			auto e = skip_range_whitespace(ctx, false);
			if (!e) { return e; }

			auto is_space_pred = detail::make_is_space_predicate(
			  ctx.locale(), (common_options & localized) != 0, field_width);
			std::string tmp;
			auto ret = do_scan(ctx, tmp, is_space_pred);
			val		 = librapid::mpq(tmp);
			return ret;
		}
	};

	template<>
	struct scanner<librapid::mpfr> : public detail::string_scanner {
		template<typename Context>
		error scan(librapid::mpfr &val, Context &ctx) {
			if (set_parser.enabled()) {
				bool loc = (common_options & localized) != 0;
				bool mb =
				  (loc || set_parser.get_option(detail::set_parser_type::flag::use_ranges)) &&
				  detail::is_multichar_type(typename Context::char_type {});
				std::string tmp;
				auto ret = do_scan(ctx, tmp, pred<Context> {ctx, set_parser, loc, mb});
				val		 = librapid::mpfr(tmp);
				return ret;
			}

			auto e = skip_range_whitespace(ctx, false);
			if (!e) { return e; }

			auto is_space_pred = detail::make_is_space_predicate(
			  ctx.locale(), (common_options & localized) != 0, field_width);
			std::string tmp;
			auto ret = do_scan(ctx, tmp, is_space_pred);
			val		 = librapid::mpfr(tmp);
			return ret;
		}
	};

	SCN_END_NAMESPACE
} // namespace scn

#	endif // SCN_SCN_H
#endif	   // LIBRAPID_USE_MULTIPREC

#endif // LIBRAPID_MATH_MULTIPREC_HPP