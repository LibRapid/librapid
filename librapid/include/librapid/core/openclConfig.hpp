#ifndef LIBRAPID_CORE_OPENCL_CONFIG_HPP
#define LIBRAPID_CORE_OPENCL_CONFIG_HPP

#if defined(LIBRAPID_HAS_OPENCL)

#    if defined(LIBRAPID_APPLE)
#        include <OpenCL/opencl.h>
#    else
#        include <CL/cl.hpp>
#    endif // LIBRAPID_APPLE

#    include <clblast.h>

#else // LIBRAPID_HAS_OPENCL

namespace librapid::typetraits {
    template<typename T>
    struct IsOpenCLStorage : std::false_type {};
} // namespace librapid::typetraits

#endif // LIBRAPID_HAS_OPENCL
#endif // LIBRAPID_CORE_OPENCL_CONFIG_HPP