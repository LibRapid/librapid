#if defined(LIBRAPID_HAS_CUDA)

#	include <librapid/librapid.hpp>

namespace librapid::cuda {
	std::string loadKernel(const std::string &kernelName) {
		static std::map<std::string, std::string> mapping;

		if (mapping.find(kernelName) != mapping.end()) { return mapping[kernelName]; }

		auto basePath = fmt::format("{}/include/librapid/cuda/kernels/", LIBRAPID_SOURCE);

		std::string helperPath = fmt::format("{}/kernelHelper.cuh", basePath);
		std::string kernelPath = fmt::format("{}/{}.cu", basePath, kernelName);
		std::fstream helper(helperPath);
		std::fstream kernel(kernelPath);
		LIBRAPID_ASSERT(helper.is_open(), "Failed to load CUDA helper kernel");
		LIBRAPID_ASSERT(kernel.is_open(), "Failed to load CUDA kernel '{}.cu'", kernelName);
		std::stringstream buffer;
		buffer << helper.rdbuf();
		buffer << "\n\n";
		buffer << kernel.rdbuf();
		helper.close();
		kernel.close();

		mapping[kernelName] = kernelName + "\n" + buffer.str();
		return mapping[kernelName];
	}
} // namespace librapid::cuda

#endif // LIBRAPID_HAS_CUDA
