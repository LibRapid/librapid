#pragma once

#include "../../internal/config.hpp"

namespace librapid::detail {
	template<typename T, typename std::enable_if_t<!std::is_floating_point_v<T>, int> = 0>
	LR_INLINE std::string kernelFormat(const T &val) {
		return fmt::format("{}", val);
	}

	template<typename T, typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
	LR_INLINE std::string kernelFormat(const T &val) {
		return fmt::format("{:.50f}", val);
	}

	template<>
	LR_INLINE std::string kernelFormat(const extended::float16_t &val) {
		return fmt::format("__half({:.50f})", val);
	}
} // namespace librapid::detail
