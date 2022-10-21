#ifndef LIBRAPID_CORE_CUDA_CONFIG_HPP
#define LIBRAPID_CORE_CUDA_CONFIG_HPP

// CUDA enabled LibRapid
#ifdef LIBRAPID_HAS_CUDA

#	ifdef _MSC_VER
// Disable warnings about unsafe classes
#		pragma warning(disable : 4996)

// Disable zero division errors
#		pragma warning(disable : 4723)
#	endif

#	define CUDA_NO_HALF // Ensure the cuda_helpers "half" data type is not defined
#	include <cublas_v2.h>
#	include <cuda.h>
#	include <curand.h>
#	include <curand_kernel.h>
#	include "../vendor/jitify/jitify.hpp"

// cuBLAS API errors
static const char *getCublasErrorEnum_(cublasStatus_t error) {
	switch (error) {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "UNKNOWN ERROR";
}

//********************//
// cuBLAS ERROR CHECK //
//********************//

#	if !defined(cublasSafeCall)
#		define cublasSafeCall(err)                                                                \
			LR_ASSERT_ALWAYS(                                                                      \
			  (err) == CUBLAS_STATUS_SUCCESS, "cuBLAS error: {}", getCublasErrorEnum_(err))
#	endif

//********************//
//  CUDA ERROR CHECK  //
//********************//

#	if defined(LIBRAPID_ENABLE_ASSERT)
#		define cudaSafeCall(call)                                                                 \
			LR_ASSERT(!(call), "CUDA Assertion Failed: {}", cudaGetErrorString(call))

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

#		define jitifyCall(call) (call)
#	endif

#	ifdef _MSC_VER
#		pragma warning(default : 4996)
#	endif

#	include "../cuda/helper_cuda.h"
#	include "../cuda/helper_functions.h"

#else

#endif // LIBRAPID_HAS_CUDA

namespace librapid::device {
	// Signifies that host memory should be used
	struct CPU {};

	// Signifies that device memory should be used
	struct GPU {};
} // namespace librapid::device

#endif // LIBRAPID_CORE_CUDA_CONFIG_HPP