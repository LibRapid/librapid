#ifndef LIBRAPID_CORE_CUDA_CONFIG_HPP
#define LIBRAPID_CORE_CUDA_CONFIG_HPP

// CUDA enabled LibRapid
#ifdef LIBRAPID_HAS_CUDA

// Under MSVC, supress a few warnings
#    ifdef _MSC_VER
#        pragma warning(push)
#        pragma warning(disable : 4505) // unreferenced local function has been removed
#    endif

#    define CUDA_NO_HALF // Ensure the cuda_helpers "half" data type is not defined

#    include <cuda.h>
#    include <cublas_v2.h>
#    include <cublasLt.h>
#    include <cublasXt.h>
#    include <curand.h>
#    include <curand_kernel.h>
#    include <cufft.h>
#    include <cufftw.h>
#    include <cuda_runtime.h>
#    include <device_launch_parameters.h>

#    ifdef _MSC_VER
#        pragma warning(pop)
#    endif

#    include "../vendor/jitify/jitify.hpp"

// cuBLAS API errors
const char *getCublasErrorEnum_(cublasStatus_t error);

//********************//
// cuBLAS ERROR CHECK //
//********************//

#    if !defined(cublasSafeCall)
#        define cublasSafeCall(err)                                                                \
            LIBRAPID_ASSERT_ALWAYS(                                                                \
              (err) == CUBLAS_STATUS_SUCCESS, "cuBLAS error: {}", getCublasErrorEnum_(err))
#    endif

//********************//
//  CUDA ERROR CHECK  //
//********************//

#    if defined(LIBRAPID_ENABLE_ASSERT)
#        define cudaSafeCall(call)                                                                 \
            LIBRAPID_ASSERT(!(call), "CUDA Assertion Failed: {}", cudaGetErrorString(call))

#        define jitifyCall(call)                                                                   \
            do {                                                                                   \
                if ((call) != CUDA_SUCCESS) {                                                      \
                    const char *str;                                                               \
                    cuGetErrorName(call, &str);                                                    \
                    throw std::runtime_error(std::string("CUDA JIT failed: ") + str);              \
                }                                                                                  \
            } while (0)
#    else
#        define cudaSafeCall(call) (call)
#        define jitifyCall(call)   (call)
#    endif

#    ifdef _MSC_VER
#        pragma warning(default : 4996)
#    endif

#    include "../cuda/helper_cuda.h"
#    include "../cuda/helper_functions.h"

#    define CONCAT_IMPL(x, y) x##y
#    define CONCAT(x, y)      CONCAT_IMPL(x, y)

#    if LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH > 1
#        define CUDA_FLOAT_VECTOR_TYPE CONCAT(jitify::float, LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH)
#    else
#        define CUDA_FLOAT_VECTOR_TYPE float
#    endif

#    if LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH > 1
#        define CUDA_DOUBLE_VECTOR_TYPE CONCAT(jitify::double, LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH)
#    else
#        define CUDA_DOUBLE_VECTOR_TYPE double
#    endif

#endif // LIBRAPID_HAS_CUDA

#endif // LIBRAPID_CORE_CUDA_CONFIG_HPP