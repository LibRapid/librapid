#ifndef LIBRAPID_CUDA_KERNEL_HELPER
#define LIBRAPID_CUDA_KERNEL_HELPER

#define LIBRAPID_INLINE inline
#define LIBRAPID_ALWAYS_INLINE inline
#define LIBRAPID_NODISCARD

#define LIBRAPID_IN_JITIFY

#include <cstdint>
#include <cuda_fp16.h>

namespace librapid {
	using half = __half;
}

#endif // LIBRAPID_CUDA_KERNEL_HELPER