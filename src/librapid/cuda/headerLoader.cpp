#include "headerLoader.hpp"

namespace librapid {
	bool loadCustomCudaHeader(const std::string &fileName, const std::string &searchDir,
							  std::vector<std::string> &res, std::vector<std::string> &args) {
		res.emplace_back(searchDir + "/" + fileName);
		args.emplace_back("-I" + searchDir);
		return true;
	}
} // namespace librapid
