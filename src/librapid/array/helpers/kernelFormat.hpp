#pragma once

#include "../../internal/config.hpp"

namespace librapid { namespace detail {
	template<typename T>
	std::string kernelFormat(const T &val) {
		return fmt::format("{}", val);
	}
} }
