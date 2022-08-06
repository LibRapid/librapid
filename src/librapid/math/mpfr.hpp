#pragma once

#if defined(LIBRAPID_USE_MULTIPREC)

#	include "../internal/forward.hpp"

// MPIR (modified) for BigNumber types
#	include <mpirxx.h>
#	include <mpreal.h>

#	include <cstdint>
#	include <thread>
#	include <future>
#	include <iostream>

namespace librapid {
	using mpz  = mpz_class;
	using mpf  = mpf_class;
	using mpq  = mpq_class;
	using mpfr = mpfr::mpreal;
	using mpc  = std::complex<mpfr>;

	std::string str(const mpz &val, const StrOpt &options = DEFAULT_STR_OPT);
	std::string str(const mpf_class &val, const StrOpt &options = DEFAULT_STR_OPT);
	std::string str(const mpq &val, const StrOpt &options = DEFAULT_STR_OPT);
	std::string str(const mpfr &val, const StrOpt &options = DEFAULT_STR_OPT);

	// TODO: Optimise these functions
	mpz toMpz(const mpz &other);
	mpz toMpz(const mpf &other);
	mpz toMpz(const mpq &other);
	mpz toMpz(const mpfr &other);

	mpz toMpf(const mpz &other);
	mpz toMpf(const mpf &other);
	mpz toMpf(const mpq &other);
	mpz toMpf(const mpfr &other);

	mpq toMpq(const mpz &other);
	mpq toMpq(const mpf &other);
	mpq toMpq(const mpq &other);
	mpq toMpq(const mpfr &other);

	mpfr toMpfr(const mpz &other);
	mpfr toMpfr(const mpf &other);
	mpfr toMpfr(const mpq &other);
	mpfr toMpfr(const mpfr &other);

	inline void prec(int64_t dig10) {
		int64_t dig2 = (int64_t)((double)dig10 * 3.32192809488736234787) + 5;
		mpf_set_default_prec(dig2);
		mpfr::mpreal::set_default_prec((mpfr_prec_t)dig2);
	}

	// Trigonometric Functionality for mpf
	mpfr sin(const mpfr &val);
	mpfr cos(const mpfr &val);
	mpfr tan(const mpfr &val);

	mpfr asin(const mpfr &val);
	mpfr acos(const mpfr &val);
	mpfr atan(const mpfr &val);
	mpfr atan2(const mpfr &dy, const mpfr &dx);

	mpfr csc(const mpfr &val);
	mpfr sec(const mpfr &val);
	mpfr cot(const mpfr &val);

	mpfr acsc(const mpfr &val);
	mpfr asec(const mpfr &val);
	mpfr acot(const mpfr &val);

	mpfr sinh(const mpfr &val);
	mpfr cosh(const mpfr &val);
	mpfr tanh(const mpfr &val);

	mpfr asinh(const mpfr &val);
	mpfr acosh(const mpfr &val);
	mpfr atanh(const mpfr &val);

	mpfr csch(const mpfr &val);
	mpfr sech(const mpfr &val);
	mpfr coth(const mpfr &val);

	mpfr acsch(const mpfr &val);
	mpfr asech(const mpfr &val);
	mpfr acoth(const mpfr &val);

	mpfr abs(const mpfr &val);
	mpfr sqrt(const mpfr &val);
	mpfr pow(const mpfr &base, const mpfr &pow);
	mpfr exp(const mpfr &val);
	mpfr exp2(const mpfr &val);
	mpfr exp10(const mpfr &val);
	mpfr ldexp(const mpfr &val, int exponent);
	mpfr log(const mpfr &val);
	mpfr log(const mpfr &val, const mpfr &base);
	mpfr log2(const mpfr &val);
	mpfr log10(const mpfr &val);

	mpfr floor(const mpfr &val);
	mpfr ceil(const mpfr &val);

	mpfr mod(const mpfr &val, const mpfr &mod);

	mpfr hypot(const mpfr &a, const mpfr &b);

	LR_INLINE mpfr constPi() { return ::mpfr::const_pi(); }
	LR_INLINE mpfr constEuler() { return ::mpfr::const_euler(); }
	LR_INLINE mpfr constLog2() { return ::mpfr::const_log2(); }
	LR_INLINE mpfr constCatalan() { return ::mpfr::const_catalan(); }
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
			LR_ASSERT("Invalid Format Specifier: {}", e.what());
			return fmt::format_to(ctx.out(), "FORMAT ERROR");
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
			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision < 1 ? 10 : specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			LR_ASSERT("Invalid Format Specifier: {}", e.what());
			return fmt::format_to(ctx.out(), "FORMAT ERROR");
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
			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision < 1 ? 10 : specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			LR_ASSERT("Invalid Format Specifier: {}", e.what());
			return fmt::format_to(ctx.out(), "FORMAT ERROR");
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
			std::stringstream ss;
			ss << std::fixed;
			ss.precision(specs_.precision < 1 ? 10 : specs_.precision);
			ss << num;
			return fmt::format_to(ctx.out(), ss.str());
		} catch (std::exception &e) {
			LR_ASSERT("Invalid Format Specifier: {}", e.what());
			return fmt::format_to(ctx.out(), fmt::format("Format Error: {}", e.what()));
		}
	}
};
#	endif // FMT_API

#else

namespace librapid {
	LR_INLINE void prec(int64_t) {};
}

#endif // LIBRAPID_USE_MULTIPREC
