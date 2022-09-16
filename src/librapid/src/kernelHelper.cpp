#include <librapid/array/helpers/kernelFormat.hpp>

namespace librapid::detail {
	std::string kernelGenerator(const std::string &opKernel,
								const std::vector<std::string> &headerFiles) {
#if defined(LIBRAPID_HAS_CUDA)
		std::string headers;
		for (const auto &header : headerFiles) {
			headers += fmt::format("#include \"{}\"\n", header);
		}

		std::string kernel = fmt::format(R"V0G0N(kernelOp
#include <stdint.h>
// Required for float16 support
#include "{0}"
// Headers
{1}

// Custom code
{2}

// Provided kernel
{3}
				)V0G0N",
										 CUDA_INCLUDE_DIRS "/cuda_fp16.h", // 0
										 headers,						   // 1
										 customCudaCode,				   // 2
										 opKernel);						   // 3

		return kernel;
#else
		return "";
#endif
	}
} // namespace librapid::detail
