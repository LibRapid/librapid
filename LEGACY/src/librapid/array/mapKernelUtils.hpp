
// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "map_kernel_interface_generator.py" in the same directory     //
// ====================================================== //

#pragma once

#include <cstdint>
#include <librapid/config.hpp>

namespace librapid::utils {
	template<typename T, typename = void>
	struct HasName : std::false_type {};

	template<typename T>
	struct HasName<T, decltype((void)T::name, void())> : std::true_type {};

	template<typename T, typename = void>
	struct HasKernel : std::false_type {};

	template<typename T>
	struct HasKernel<T, decltype((void)T::kernel, void())> : std::true_type {};

	template<typename T, typename Kernel, uint64_t dims>
	struct ApplyKernelImpl {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			throw std::runtime_error(
			  "Too many arguments passed to Array.mapKernel -- "
			  "Please see the documentation for details");
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 1> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 2> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index], pointers[1][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 3> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(
			  pointers[0][index], pointers[1][index], pointers[2][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 4> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 5> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 6> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 7> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 8> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 9> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 10> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index],
								pointers[9][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 11> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index],
								pointers[9][index],
								pointers[10][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 12> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index],
								pointers[9][index],
								pointers[10][index],
								pointers[11][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 13> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index],
								pointers[9][index],
								pointers[10][index],
								pointers[11][index],
								pointers[12][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 14> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index],
								pointers[9][index],
								pointers[10][index],
								pointers[11][index],
								pointers[12][index],
								pointers[13][index]);
		}
	};

	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, 15> {
		static inline void run(T **__restrict pointers, T *__restrict dst,
							   const Kernel &kernel, uint64_t index) {
			dst[index] = kernel(pointers[0][index],
								pointers[1][index],
								pointers[2][index],
								pointers[3][index],
								pointers[4][index],
								pointers[5][index],
								pointers[6][index],
								pointers[7][index],
								pointers[8][index],
								pointers[9][index],
								pointers[10][index],
								pointers[11][index],
								pointers[12][index],
								pointers[13][index],
								pointers[14][index]);
		}
	};

} // namespace librapid::utils