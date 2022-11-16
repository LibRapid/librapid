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
	std::string str(const mpz &val, int base = 10);

	/// Convert a multiprecision floating point type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpf &val, int base = 10);

	/// Convert a multiprecision rational type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpq &val, int base = 10);

	/// Convert a multiprecision floating point type to a string with a given base
	/// \param val The value to convert
	/// \param base The base to convert to
	/// \return The converted value
	std::string str(const mpfr &val, int base = 10);

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

	/// Set the precision for multiprecision types
	/// \param dig10
	inline void prec(int64_t dig10) {
		int64_t dig2 = (int64_t)((double)dig10 * 3.32192809488736234787) + 5;
		mpf_set_default_prec(dig2);
		mpfr::mpreal::set_default_prec((mpfr_prec_t)dig2);
	}

	/// Set the precision for multiprecision types
	/// \param dig2
	inline void prec2(int64_t dig2) {
		dig2 += 8; // Add some extra precision
		mpf_set_default_prec(dig2);
		mpfr::mpreal::set_default_prec((mpfr_prec_t)dig2);
	}

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
	/// arccotangent of the value
	mpfr acoth(const mpfr &val);

	/// Absolute value of a multiprecision floating point value: \f$ |x| \f$
	/// \param val The value to take the absolute value of
	/// \return The absolute value of the value
	mpfr abs(const mpfr &val);

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

	/// Calculate and return \f$ e \f$ with LibRapid's current precision
	/// \return \f$ e \f$
	/// \see prec
	LIBRAPID_ALWAYS_INLINE mpfr constEuler() { return ::mpfr::const_euler(); }

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
			ss.precision(specs_.precision < 1 ? 10 : specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};

// Even though `mpf` is not a typedef-ed type, we'll add printing support for it
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

#else

namespace librapid {
	LIBRAPID_ALWAYS_INLINE void prec(int64_t) {};
}

#endif // LIBRAPID_USE_MULTIPREC

#endif // LIBRAPID_MATH_MULTIPREC_HPP