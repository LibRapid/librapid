#include <librapid/cuda/cudaCodeLoader.hpp>

#if defined(HAVE_CONFIG_H)
#	error "WHY IS C++ SO AWFUL"
#endif

namespace librapid {
	void loadCustomCudaHeader(const std::string &fileName, const std::string &searchDir,
							  std::vector<std::string> &res, std::vector<std::string> &args) {
		res.emplace_back(searchDir + "/" + fileName);
		args.emplace_back("-I" + searchDir);
	}

	void registerCudaCode(const std::string &code, std::string &dst) {
		dst += "\n\n\n// NEW CODE BLOCK\n" + code + "\n\n\n";
	}
} // namespace librapid
