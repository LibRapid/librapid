#pragma once

// Provide lrc::str(x) for all LibRapid types and primitive types. This acts as a helper in most
// cases but can also be extended by other libraries to provide easier formatting.

#include "../internal/config.hpp"
#include "../internal/forward.hpp"

namespace librapid {
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	inline std::string str(const T &val, const StrOpt &options = {-1, 10, false}) {
		return fmt::format("{}", val);
	}
} // namespace librapid