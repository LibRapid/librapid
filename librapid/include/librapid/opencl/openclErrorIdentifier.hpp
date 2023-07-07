#ifndef LIBRAPID_OPENCL_OPENCL_ERROR_IDENTIFIER_HPP
#define LIBRAPID_OPENCL_OPENCL_ERROR_IDENTIFIER_HPP

namespace librapid::opencl {
	std::string getOpenCLErrorString(int64_t err);
	std::string getCLBlastErrorString(clblast::StatusCode err);
} // namespace librapid::opencl

#endif // LIBRAPID_OPENCL_OPENCL_ERROR_IDENTIFIER_HPP