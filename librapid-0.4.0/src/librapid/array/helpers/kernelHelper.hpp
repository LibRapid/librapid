#pragma once

#include "../../internal/config.hpp"

namespace librapid::detail {
	std::string kernelGenerator(const std::string &opKernel,
								const std::vector<std::string> &headerFiles = customHeaders);
} // namespace librapid::detail