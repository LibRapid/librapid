#ifndef LIBRAPID_CORE_OPENCL_CONFIG_HPP
#define LIBRAPID_CORE_OPENCL_CONFIG_HPP

#if defined(LIBRAPID_HAS_OPENCL)

// Required files...
#include <complex>

#	if defined(__APPLE__)
// 		On MacOS, we have to use custom C++ bindings, since they are not provided by default
#		include <OpenCL/opencl.h>
#		include <librapid/opencl/opencl.hpp>
#	else
#		include <CL/cl.hpp>
#	endif // LIBRAPID_APPLE

#	include <clblast.h>

#else // LIBRAPID_HAS_OPENCL

namespace librapid::typetraits {
	template<typename T>
	struct IsOpenCLStorage : std::false_type {};
} // namespace librapid::typetraits

#endif // LIBRAPID_HAS_OPENCL
#endif // LIBRAPID_CORE_OPENCL_CONFIG_HPP