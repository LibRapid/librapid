#pragma once

#include "../internal/config.hpp"

namespace librapid {
	bool loadCustomCudaHeader(const std::string &fileName, const std::string &searchDir = "",
							  std::vector<std::string> &res	 = customHeaders,
							  std::vector<std::string> &args = nvccOptions);
}
