#pragma once

#if defined(LIBRAPID_HAS_CUDA)
#	include "../internal/config.hpp"
#	include "cudaCodeLoader.hpp"
#	include "../internal/memUtils.hpp"

// Memory alignment adapted from
// https://gist.github.com/dblalock/255e76195676daa5cbc57b9b36d1c99a

namespace librapid::memory {
	static bool streamCreated = false;
	static cudaStream_t cudaStream;
	static constexpr int64_t handleBufferSize = LIBRAPID_MAX_ALLOWED_THREADS * LIBRAPID_OMP_VAL;
	static int64_t handleSize				  = handleBufferSize;
	static cublasHandle_t cublasHandles[handleBufferSize];

	LR_INLINE void initializeCudaStream() {
#	ifdef LIBRAPID_HAS_CUDA
		if (!streamCreated) {
			checkCudaErrors(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));

			handleSize = std::thread::hardware_concurrency();
#		if !defined(LIBRAPID_NO_THREAD_CHECK)
			if (handleSize > 256) {
				fmt::print("\n{:=>90}\n", "");
				fmt::print(R"V0G0N(
Hello! You seem to have a LOT of threads (256+) and a CUDA-enabled GPU. This could cause
problems with concurrency and CUDA streams, so LibRapid has a buffer of 256 (maximum)
cublasHandle_t instances. If you are not running LibRapid code in parallel (outside of
LibRapid's own processing), you don't need to worry about this, however it is STRONGLY
advised that you don't ignore it...

If you're using C++, simply define "LIBRAPID_MAX_ALLOWED_THREADS" before including
LibRapid. If you're using Python, you'll need to define "LIBRAPID_MAX_ALLOWED_THREADS"
in "./librapidPythonInterface.cpp" above the include statement and then rebuild the
library.

Instructions to rebuild the Python library can be found in the Documentation at
"http://librapid.rtfd.io/", but the simplest way to install the library is to simply
run "pip install ." from within the LibRapid directory. (You may also want to add
"-vvv" to the command as it can take a while, and this way you get an idea of how far
through it is)

To IGNORE this error, just define LIBRAPID_NO_THREAD_CHECK above LibRapid includes :)
)V0G0N");
				fmt::print("\n{:=>90}\n", "");
				std::exit(1);
			}
#		endif

			for (int64_t i = 0; i < handleSize; ++i) {
				cublasCreate_v2(&(cublasHandles[i]));
				cublasSetStream_v2(cublasHandles[i], cudaStream);
			}

			streamCreated = true;

			loadCustomCudaHeader("cuda_fp16.h", CUDA_INCLUDE_DIRS);
		}
#	endif // LIBRAPID_HAS_CUDA
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::GPU>, int> = 0>
	LR_NODISCARD("Do not leave a dangling pointer")
	LR_FORCE_INLINE T *malloc(size_t num, size_t alignment = memAlign, bool zero = false) {
		// Ignore memory alignment
		T *buf;
		cudaSafeCall(cudaMallocAsync(&buf, sizeof(T) * num, cudaStream));
// Slightly altered traceback call to log u_chars being allocated
#	ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- MALLOC {} u_charS -> {}", size, (void *)buf);
#	endif

		return buf;
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::GPU>, int> = 0>
	LR_FORCE_INLINE void free(T *ptr) {
#	ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- FREE {}", (void *)alignedPtr);
#	endif

		cudaSafeCall(cudaFreeAsync(ptr, cudaStream));
	}

	template<typename T, typename d, typename T_, typename d_,
			 typename std::enable_if_t<
			   !(std::is_same_v<d, device::CPU> && std::is_same_v<d_, device::CPU>), int> = 0>
	LR_FORCE_INLINE void memcpy(T *dst, T_ *src, int64_t size) {
		if constexpr (std::is_same_v<T, T_>) {
			if constexpr (std::is_same_v<d, device::CPU> && std::is_same_v<d_, device::GPU>) {
				// Device to Host
				cudaSafeCall(
				  cudaMemcpyAsync(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost, cudaStream));
			} else if constexpr (std::is_same_v<d, device::GPU> &&
								 std::is_same_v<d_, device::CPU>) {
				// Host to Device
				cudaSafeCall(
				  cudaMemcpyAsync(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice, cudaStream));
			} else if constexpr (std::is_same_v<d, device::GPU> &&
								 std::is_same_v<d_, device::GPU>) {
				// Host to Device
				cudaSafeCall(cudaMemcpyAsync(
				  dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice, cudaStream));
			}
		} else {
			// TODO: Optimise this

			if constexpr (std::is_same_v<d_, device::CPU>) {
				// Source device is CPU
				for (int64_t i = 0; i < size; ++i) {
					T tmp = src[i]; // Required to cast value

					// Copy from host to device
					cudaSafeCall(cudaMemcpyAsync(
					  dst + i, &tmp, sizeof(T), cudaMemcpyHostToDevice, cudaStream));
				}
			} else if constexpr (std::is_same_v<d, device::CPU>) {
				// Destination device is CPU
				for (int64_t i = 0; i < size; ++i) {
					T_ tmp; // Required to cast value

					// Copy from device to host
					cudaSafeCall(cudaMemcpyAsync(
					  &tmp, src + i, sizeof(T_), cudaMemcpyDeviceToHost, cudaStream));
					dst[i] = tmp; // Write final result
				}
			} else {
				const char *kernel = R"V0G0N(memcpyKernel
					#include <stdint.h>
					template<typename DST, typename SRC>
					__global__
					void memcpyKernel(DST *dst, SRC *src, int64_t size) {
						uint64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
						if (kernelIndex < size) dst[kernelIndex] = src[kernelIndex];
					}
				)V0G0N";

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel);
				unsigned int threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (size < 512) {
					threadsPerBlock = (unsigned int)size;
					blocksPerGrid	= 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid	= ceil(double(size) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				using jitify::reflection::Type;
				jitifyCall(program.kernel("memcpyKernel")
							 .instantiate(Type<T>(), Type<T_>())
							 .configure(grid, block, 0, cudaStream)
							 .launch(dst, src, size));
			}
		}
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::GPU>, int> = 0>
	LR_FORCE_INLINE void memset(T *dst, int val, int64_t size) {
		cudaMemsetAsync((void *) dst, val, sizeof(T) * size, cudaStream);
	}
} // namespace librapid::memory
#endif // LIBRAPID_HAS_CUDA