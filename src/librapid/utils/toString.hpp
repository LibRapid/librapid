#pragma once

// Provide lrc::str(x) for all LibRapid types and primitive types. This acts as a helper in most
// cases but can also be extended by other libraries to provide easier formatting.

#include "../internal/config.hpp"

namespace librapid {
	struct StrOpt {
		int64_t digits = -1;
		int64_t base = 10;
		bool scientific = false;
	};

	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	inline std::string str(const T &val, const StrOpt &options = {-1, 10, false}) {
		return fmt::format("{}", val);
	}

	template<typename T, typename D>
	inline std::string str(const Array<T, D> &val, const StrOpt &options = {-1, 10, false}) {
		return val.str();
	}

	template<typename T, int64_t d, int64_t a>
	inline std::string str(const ExtentType<T, d, a> &val, const StrOpt &options = {-1, 10, false}) {
		return val.str();
	}

	std::string str(const mpz &val, const StrOpt &options = {-1, 10, false});
	std::string str(const mpf &val, const StrOpt &options = {-1, 10, false});
	std::string str(const mpq &val, const StrOpt &options = {-1, 10, false});
} // namespace librapid