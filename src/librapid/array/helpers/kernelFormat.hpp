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
		return fmt::format(
		  "(({}) {})", internal::traits<typename internal::traits<T>::BaseScalar>::Name, val);
	}

	std::string kernelGenerator(const std::string &opKernel,
								const std::vector<std::string> &headerFiles = customHeaders);
} // namespace librapid::detail
