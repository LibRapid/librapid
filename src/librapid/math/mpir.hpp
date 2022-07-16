#pragma once

#if defined(LIBRAPID_USE_MPIR)

// MPIR (modified) for BigNumber types
#	include <cstdint>
#	include <mpirxx.h>
#	include <thread>
#	include <future>
#	include <iostream>

namespace librapid {
	using mpz = mpz_class;
	using mpf = mpf_class;
	using mpq = mpq_class;

	inline void prec(int64_t dig10) { mpf_set_default_prec((int64_t)((double)dig10 * 3.33)); }

	namespace detail {
		struct PQT {
			mpz_class P, Q, T;
		};
	} // namespace detail

	class Chudnovsky {
	public:
		explicit Chudnovsky(int64_t dig10 = 100);
		[[nodiscard]] detail::PQT compPQT(int32_t n1, int32_t n2) const;
		[[nodiscard]] mpf pi() const;

		[[nodiscard]] mpf piMultiThread() const;

		[[nodiscard]] static detail::PQT compPQT2(const Chudnovsky &chud, int32_t n1, int32_t n2,
												  int64_t depth = 0) {
			int32_t m;
			detail::PQT res;

			if (n1 + 1 == n2) {
				res.P = mpz(2 * n2 - 1);
				res.P *= (6 * n2 - 1);
				res.P *= (6 * n2 - 5);
				res.Q = chud.C3_24 * n2 * n2 * n2;
				res.T = (chud.A + chud.B * n2) * res.P;
				if ((n2 & 1) == 1) res.T = -res.T;
			} else {
				// I'll be honest: I have no clue why this works. Theoretically it should be slower
				// than other methods, but it seems to be faster, so I'm not complaining :)
				auto maxThreads = std::thread::hardware_concurrency() / 2;
				if (depth < maxThreads) {
					m = (n1 + n2) / 2;

					std::future<detail::PQT> res1fut =
					  std::async(&Chudnovsky::compPQT2, chud, n1, m, depth + 1);

					std::future<detail::PQT> res2fut =
					  std::async(&Chudnovsky::compPQT2, chud, m, n2, depth + 1);

					detail::PQT res1 = res1fut.get(); // compPQT2(n1, m);
					detail::PQT res2 = res2fut.get(); // compPQT2(m, n2);
					res.P			 = res1.P * res2.P;
					res.Q			 = res1.Q * res2.Q;
					res.T			 = res1.T * res2.Q + res1.P * res2.T;
				} else {
					m				 = (n1 + n2) / 2;
					detail::PQT res1 = compPQT2(chud, n1, m, depth + 1);
					detail::PQT res2 = compPQT2(chud, m, n2, depth + 1);
					res.P			 = res1.P * res2.P;
					res.Q			 = res1.Q * res2.Q;
					res.T			 = res1.T * res2.Q + res1.P * res2.T;
				}
			}

			return res;
		}

	public:
		mpz A, B, C, D, E, C3_24;
		int64_t DIGITS, PREC, N;
		double DIGITS_PER_TERM;
	};

	mpf epsilon(const mpf &val = mpf_class());
	mpf fmod(const mpf &val, const mpf &mod);

	// Trigonometric Functionality for mpf
	mpf sin(const mpf &val);
	mpf cos(const mpf &val);
	mpf tan(const mpf &val);

	mpf csc(const mpf &val);
	mpf sec(const mpf &val);
	mpf cot(const mpf &val);

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
#	endif // FMT_API

#endif // LIBRAPID_USE_MPIR
