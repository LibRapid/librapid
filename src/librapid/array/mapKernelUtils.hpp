#pragma once

#include <librapid/config.hpp>
#include <cstdint>

#define KERNEL_EXPAND_1(ptrs, dst, index) dst[index] = kernel(ptrs[0])

namespace librapid::utils {
	template<typename T, typename Kernel, uint64_t dims>
	struct ApplyKernelImpl {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 1> {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 2> {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index], pointers[1][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 3> {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index], pointers[1][index], pointers[2][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 4> {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index], pointers[1][index], pointers[2][index], pointers[3][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 5> {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index], pointers[1][index], pointers[2][index], pointers[3][index], pointers[4][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 6> {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index], pointers[1][index], pointers[2][index], pointers[3][index], pointers[4][index], pointers[5][index]);
		}
	};
}
