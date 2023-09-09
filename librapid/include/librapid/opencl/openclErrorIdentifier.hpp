#ifndef LIBRAPID_OPENCL_OPENCL_ERROR_IDENTIFIER_HPP
#define LIBRAPID_OPENCL_OPENCL_ERROR_IDENTIFIER_HPP

namespace librapid::opencl {
#if defined(LIBRAPID_HAS_OPENCL)
    std::string getOpenCLErrorString(int64_t err);
    std::string getCLBlastErrorString(clblast::StatusCode err);
#endif // LIBRAPID_HAS_OPENCL
} // namespace librapid::opencl

#endif // LIBRAPID_OPENCL_OPENCL_ERROR_IDENTIFIER_HPP