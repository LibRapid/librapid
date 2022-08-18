#pragma once

#include "../../internal/config.hpp"
#include "../../utils/traits.hpp"

namespace librapid::detail {
	template<typename T, typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
	LR_INLINE std::string kernelFormat(const T &val) {
		return fmt::format("(({}) {:.50f})", internal::traits<T>::Name, val);
	}

	template<typename T, typename std::enable_if_t<!std::is_floating_point_v<T>, int> = 0>
	LR_INLINE std::string kernelFormat(const T &val) {
		return fmt::format("(({}) {})", internal::traits<T>::Name, val);
	}
} // namespace librapid::detail
