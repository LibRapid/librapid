#ifndef LIBRAPID_CUDA_KERNEL_HELPER
#define LIBRAPID_CUDA_KERNEL_HELPER

#include <cstdint>

// If compiling with NVCC, disable all macros, otherwise enable them so the IDE is happy
#if defined(__NVCC__)
#	define HAS_NVCC
#endif

#if !defined(HAS_NVCC)
#	define __global__
#	define __device__
#	define __host__
#	define __constant__
#	define __launch_bounds__
#	define __shared__
#	define __global
#	define __device
#	define __constant__

struct dim3 {
	unsigned x;
    unsigned y;
    unsigned z;
};

#define blockDim (dim3{0, 0, 0})
#define blockIdx (dim3{0, 0, 0})
#define threadIdx (dim3{0, 0, 0})
#endif // HAS_NVCC

#endif // LIBRAPID_CUDA_KERNEL_HELPER