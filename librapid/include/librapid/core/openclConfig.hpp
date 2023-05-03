#ifndef LIBRAPID_CORE_OPENCL_CONFIG_HPP
#define LIBRAPID_CORE_OPENCL_CONFIG_HPP

#if defined(LIBRAPID_HAS_OPENCL)

#	if defined(LIBRAPID_APPLE)
#		include <OpenCL/cl.hpp>
#	else
#		include <CL/cl.hpp>
#	endif // LIBRAPID_APPLE



#endif	   // LIBRAPID_HAS_OPENCL
#endif	   // LIBRAPID_CORE_OPENCL_CONFIG_HPP