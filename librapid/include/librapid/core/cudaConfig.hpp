#ifndef LIBRAPID_CORE_CUDA_CONFIG_HPP
#define LIBRAPID_CORE_CUDA_CONFIG_HPP

// CUDA enabled LibRapid
#ifdef LIBRAPID_HAS_CUDA

// Under MSVC, supress a few warnings
#	ifdef _MSC_VER
#		pragma warning(push)
#		pragma warning(disable : 4505) // unreferenced local function has been removed
#	endif

#	define CUDA_NO_HALF // Ensure the cuda_helpers "half" data type is not defined
#	include <cuda.h>
#	include <cublas_v2.h>
#	include <curand.h>
#	include <curand_kernel.h>
#	include <cufft.h>
#	include <cufftw.h>
#	include <cuda_runtime.h>
#	include <device_launch_parameters.h>

#	ifdef _MSC_VER
#		pragma warning(pop)
#	endif

#	include "../vendor/jitify/jitify.hpp"

// cuBLAS API errors
const char *getCublasErrorEnum_(cublasStatus_t error);

//********************//
// cuBLAS ERROR CHECK //
//********************//

#	if !defined(cublasSafeCall)
#		define cublasSafeCall(err)                                                                \
			LIBRAPID_ASSERT_ALWAYS(                                                                \
			  (err) == CUBLAS_STATUS_SUCCESS, "cuBLAS error: {}", getCublasErrorEnum_(err))
#	endif

//********************//
//  CUDA ERROR CHECK  //
//********************//

#	if defined(LIBRAPID_ENABLE_ASSERT)
#		define cudaSafeCall(call)                                                                 \
			LIBRAPID_ASSERT(!(call), "CUDA Assertion Failed: {}", cudaGetErrorString(call))

#		define jitifyCall(call)                                                                   \
			do {                                                                                   \
				if ((call) != CUDA_SUCCESS) {                                                      \
					const char *str;                                                               \
					cuGetErrorName(call, &str);                                                    \
					throw std::runtime_error(std::string("CUDA JIT failed: ") + str);              \
				}                                                                                  \
			} while (0)
#	else
#		define cudaSafeCall(call) (call)
#		define jitifyCall(call)   (call)
#	endif

#	ifdef _MSC_VER
#		pragma warning(default : 4996)
#	endif

#	include "../cuda/helper_cuda.h"
#	include "../cuda/helper_functions.h"

#endif // LIBRAPID_HAS_CUDA

// This needs to be defined before cudaHeaderLoader.hpp is included

#include "../cuda/cudaKernelProcesor.hpp"

#endif // LIBRAPID_CORE_CUDA_CONFIG_HPP