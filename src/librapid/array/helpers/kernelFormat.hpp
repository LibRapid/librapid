#pragma once

#include "../../internal/config.hpp"

namespace librapid::detail {
	template<typename T>
	LR_INLINE std::string kernelFormat(const T &val) {
		return fmt::format("{}", val);
	}
}
