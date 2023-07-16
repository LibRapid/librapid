#ifndef LIBRAPID_CUDA_KERNEL_HELPER
#define LIBRAPID_CUDA_KERNEL_HELPER

#define LIBRAPID_INLINE inline
#define LIBRAPID_ALWAYS_INLINE inline
#define LIBRAPID_NODISCARD

#define LIBRAPID_IN_JITIFY

#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>

namespace librapid {
	using half = __half;
}

template<typename T>
struct IsHalf : std::false_type {};

template<>
struct IsHalf<librapid::half> : std::true_type {};

#endif // LIBRAPID_CUDA_KERNEL_HELPER