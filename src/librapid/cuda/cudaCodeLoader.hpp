#pragma once

#include "../internal/config.hpp"

namespace librapid {
	void loadCustomCudaHeader(const std::string &fileName, const std::string &searchDir = "",
							  std::vector<std::string> &res	 = customHeaders,
							  std::vector<std::string> &args = nvccOptions);

	void registerCudaCode(const std::string &code, std::string &dst = customCudaCode);
} // namespace librapid
