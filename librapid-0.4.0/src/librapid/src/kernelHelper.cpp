#include <librapid/array/helpers/kernelHelper.hpp>

namespace librapid::detail {
	std::string kernelGenerator(const std::string &opKernel, const std::vector<std::string> &headerFiles) {
#if defined(LIBRAPID_HAS_CUDA)
		std::string headers;
		for (const auto &header : headerFiles) {
			headers += fmt::format("#include \"{}\"\n", header);
		}

		std::string kernel = fmt::format(R"V0G0N(kernelOp
#include <stdint.h>
// Headers
{0}

// Custom code
{1}

// Provided kernel
{2}
				)V0G0N",
										 headers,		 // 0
										 customCudaCode, // 1
										 opKernel);		 // 2

		return kernel;
#else
		return "";
#endif
	}
} // namespace librapid::detail
