#if defined(LIBRAPID_HAS_CUDA)

#	include <librapid/librapid.hpp>

namespace librapid {
	std::string loadKernel(const std::string &kernelName) {
		static std::map<std::string, std::string> mapping;

		if (mapping.find(kernelName) != mapping.end()) { return mapping[kernelName]; }

		auto path =
		  fmt::format("{}/include/librapid/cuda/kernels/{}.cu", LIBRAPID_SOURCE, kernelName);
		std::fstream file(path);
		LIBRAPID_ASSERT(file.is_open(), "Failed to load CUDA kernel '{}.cu'", kernelName);
		std::stringstream buffer;
		buffer << file.rdbuf();
		file.close();

		mapping[kernelName] = kernelName + "\n" + buffer.str();
		return mapping[kernelName];
	}
} // namespace librapid

#endif // LIBRAPID_HAS_CUDA
