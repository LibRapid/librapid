#pragma once

#include "../internal/config.hpp"
#include "../array/helpers/kernelFormat.hpp"
#include "../internal/memUtils.hpp"
#include "../cuda/memUtils.hpp"
#include "../modified/modified.hpp"

namespace librapid {
	LR_INLINE void setCudaMathMode(const std::string &type) {
#if defined(LIBRAPID_HAS_CUDA)
		if (type == "fast")
			internal::cudaMathMode = "FAST";
		else if (type == "default")
			internal::cudaMathMode = "DEFAULT";
		else if (type == "precise")
			internal::cudaMathMode = "PRECISE";
		else
			LR_ASSERT(false, "Invalid CUDA math mode {}", type);
#endif
	}

	namespace blas {
		template<typename A, typename B>
		using Common = typename std::common_type_t<A, B>;

#if defined(LIBRAPID_HAS_CUDA)

		// Use the table from https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx
		LR_INLINE int64_t cudaComputeType(const cublasDataType_t &A, const cublasDataType_t &B,
										  const cublasDataType_t &C, cublasComputeType_t &res) {
			// TODO: Implement for all available types
			if (A == CUDA_R_16F && B == CUDA_R_16F && C == CUDA_R_16F) {
				if (internal::cudaMathMode == "PRECISE") {
					res = CUBLAS_COMPUTE_16F_PEDANTIC;
				} else {
					res = CUBLAS_COMPUTE_16F;
				}
				return true;
			}

			if (A == CUDA_R_32F && B == CUDA_R_32F && C == CUDA_R_32F) {
				if (internal::cudaMathMode == "PRECISE") {
					res = CUBLAS_COMPUTE_32F_FAST_TF32;
				} else {
					res = CUBLAS_COMPUTE_32F_FAST_16F;
				}
				return true;
			}

			if (A == CUDA_C_32F && B == CUDA_C_32F && C == CUDA_C_32F) {
				if (internal::cudaMathMode == "PRECISE") {
					res = CUBLAS_COMPUTE_32F_FAST_TF32;
				} else {
					res = CUBLAS_COMPUTE_32F_FAST_16F;
				}
				return true;
			}

			if (A == CUDA_R_64F && B == CUDA_R_64F && C == CUDA_R_64F) {
				if (internal::cudaMathMode == "PRECISE") {
					res = CUBLAS_COMPUTE_64F_PEDANTIC;
				} else {
					res = CUBLAS_COMPUTE_64F;
				}
				return true;
			}

			if (A == CUDA_C_64F && B == CUDA_C_64F && C == CUDA_C_64F) {
				if (internal::cudaMathMode == "PRECISE") {
					res = CUBLAS_COMPUTE_64F_PEDANTIC;
				} else {
					res = CUBLAS_COMPUTE_64F;
				}
				return true;
			}

			return false;
		}
#endif

		template<typename Device, typename A, typename B>
		Common<A, B> dot(int64_t n, A *__restrict x, int64_t incX, B *__restrict y, int64_t incY) {
			if constexpr (std::is_same_v<Device, device::CPU>) {
				Common<A, B> res(0);
				for (int64_t i = 0; i < n; i++) res += x[i * incX] * y[i * incY];
				return res;
			}
#if defined(LIBRAPID_HAS_CUDA)
			else {
				using jitify::reflection::Type;

				int64_t threadsPerBlock, blocksPerGrid;
				// Use 1 to 512 threads per block
				if (n < 512) {
					threadsPerBlock = n;
					blocksPerGrid	= 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid	= ceil(double(n) / double(threadsPerBlock));
				}

				std::string opKernel =
				  fmt::format("#define BLOCK_SIZE {}\n", threadsPerBlock) + R"V0G0N(
template<typename A, typename B, typename C>
__global__
void dot(int64_t n, A *__restrict x, int64_t incX, B *__restrict y, int64_t incY, C *__restrict dst) {
	__shared__ C cache[BLOCK_SIZE];
	int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	cache[threadIdx.x] = 0;
	while (i < n) {
		cache[threadIdx.x] += x[i * incX] * y[i * incY];
		i += gridDim.x * blockDim.x;
	}
	__syncthreads();  // required because later on the current thread is
					  // accessing data written by another thread
	i = BLOCK_SIZE / 2;
	while (i > 0) {
		if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;  // not sure bitwise operations are actually faster
	}
	#ifndef NO_SYNC  // serialized access to shared data;
		if (threadIdx.x == 0) atomicAdd(dst, cache[0]);
	#else  // no sync, what most likely happens is:
		// 1) all threads read 0
		// 2) all threads write concurrently 16 (local block dot product)
		if (threadIdx.x == 0) *dst += cache[0];
	#endif
}
)V0G0N";

				std::string kernel = detail::kernelGenerator(opKernel);

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, nvccOptions);

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				Common<A, B> *deviceRes = memory::malloc<Common<A, B>, device::GPU>(1);

				jitifyCall(program.kernel("dot")
							 .instantiate(Type<A>(), Type<B>(), Type<Common<A, B>>())
							 .configure(grid, block, 0, memory::cudaStream)
							 .launch(n, x, incX, y, incY, deviceRes));

				Common<A, B> hostRes;
				memory::memcpy<Common<A, B>, device::CPU, Common<A, B>, device::GPU>(
				  &hostRes, deviceRes, 1);
				memory::free<Common<A, B>, device::GPU>(deviceRes);

				return hostRes;
			}
#endif
		}

		template<typename Device, typename A, typename B, typename C>
		void gemm(bool transA, bool transB, int64_t m, int64_t n, int64_t k, A alpha,
				  const A *__restrict a, int64_t lda, const B *__restrict b, int64_t ldb, B beta,
				  C *__restrict c, int64_t ldc) {
			if constexpr (std::is_same_v<Device, device::CPU>) {
				// CPU implementation

				B betaZero(0);
				C cZero(0);

#if defined(LIBRAPID_HAS_OMP)
#	pragma omp parallel for shared(                                                               \
	  transA, transB, m, n, k, a, b, c, alpha, beta, lda, ldb, ldc, betaZero, cZero) default(none) \
	  schedule(static)
#endif
				for (int64_t i = 0; i < m; ++i) {
					for (int64_t j = 0; j < n; ++j) {
						C sum;
						if (beta == betaZero)
							sum = cZero;
						else
							sum = c[i * ldc + j];

						for (int64_t l = 0; l < k; ++l) {
							sum += a[transA ? i + l * lda : l + i * lda] *
								   b[transB ? l + j * ldb : j + l * ldb];
						}
						c[i * ldc + j] = sum * alpha;
					}
				}
			}
#if defined(LIBRAPID_HAS_CUDA)
			else {
				// CUDA implementation

#	if defined(LIBRAPID_HAS_OMP)
				int64_t threadId = omp_get_thread_num();
#	else
				int64_t threadId = 0;
#	endif

				cublasDataType_t typeA = internal::traits<A>::CudaType;
				cublasDataType_t typeB = internal::traits<B>::CudaType;
				cublasDataType_t typeC = internal::traits<C>::CudaType;

				cublasComputeType_t computeType;
				bool gemmEx = cudaComputeType(typeA, typeB, typeC, computeType);

				LR_ASSERT(gemmEx, "Unsupported GEMM configuration");

				cublasSafeCall(cublasGemmEx(memory::cublasHandles[threadId],
											transA ? CUBLAS_OP_T : CUBLAS_OP_N,
											transB ? CUBLAS_OP_T : CUBLAS_OP_N,
											n,
											m,
											k,
											&alpha,
											b,
											typeA,
											ldb,
											a,
											typeB,
											lda,
											&beta,
											c,
											typeC,
											ldc,
											computeType,
											CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
#endif // LIBRAPID_HAS_CUDA
		}

		template<typename Device, typename A, typename B, typename C>
		void gemv(bool trans, int64_t m, int64_t n, A alpha, const A *__restrict a, int64_t lda,
				  const B *__restrict x, int64_t incX, B beta, C *__restrict y, int64_t incY) {
			if (std::is_same_v<Device, device::CPU>) {
				// CPU implementation

				// Matrix vector product
				if (m * n < threadThreshold || matrixThreads < 2) {
					for (int64_t i = 0; i < m; ++i) {
						y[i * incY] = beta * y[i * incY];
						for (int64_t j = 0; j < n; ++j) {
							y[i * incY] +=
							  alpha * a[trans ? j * lda + i : i * lda + j] * x[j * incX];
						}
					}
				} else {
#if defined(LIBRAPID_HAS_OMP)
#	pragma omp parallel for shared(trans, m, n, alpha, a, lda, x, incX, beta, y, incY) default(   \
	  none) num_threads(matrixThreads)
#endif // LIBRAPID_HAS_OMP
					for (int64_t i = 0; i < m; ++i) {
						y[i * incY] = beta * y[i * incY];
						for (int64_t j = 0; j < n; ++j) {
							y[i * incY] +=
							  alpha * a[trans ? j * lda + i : i * lda + j] * x[j * incX];
						}
					}
				}
			}
#if defined(LIBRAPID_HAS_CUDA)
			else {
				// CUDA implementation -- call GEMM instead (supports more datatypes with GemmEx)
				gemm<Device>(trans, false, m, 1, n, alpha, a, lda, x, incX, beta, y, incY);
			}
#endif // LIBRAPID_HAS_CUDA
		}

#if defined(LIBRAPID_HAS_BLAS)
		template<>
		float dot<device::CPU, float, float>(int64_t n, float *__restrict x, int64_t incX,
											 float *__restrict y, int64_t incY);

		template<>
		double dot<device::CPU, double, double>(int64_t n, double *__restrict x, int64_t incX,
												double *__restrict y, int64_t incY);

		template<>
		void gemv<device::CPU, float, float, float>(bool trans, int64_t m, int64_t n, float alpha,
													const float *__restrict a, int64_t lda,
													const float *__restrict x, int64_t incX,
													float beta, float *__restrict y, int64_t incY);

		template<>
		void gemv<device::CPU, double, double, double>(bool trans, int64_t m, int64_t n,
													   double alpha, const double *__restrict a,
													   int64_t lda, const double *__restrict x,
													   int64_t incX, double beta,
													   double *__restrict y, int64_t incY);

		template<>
		void gemm<device::CPU, float, float, float>(bool transA, bool transB, int64_t m, int64_t n,
													int64_t k, float alpha,
													const float *__restrict a, int64_t lda,
													const float *__restrict b, int64_t ldb,
													float beta, float *__restrict c, int64_t ldc);

		template<>
		void gemm<device::CPU, double, double, double>(bool transA, bool transB, int64_t m,
													   int64_t n, int64_t k, double alpha,
													   const double *__restrict a, int64_t lda,
													   const double *__restrict b, int64_t ldb,
													   double beta, double *__restrict c,
													   int64_t ldc);
#endif

#if defined(LIBRAPID_HAS_CUDA)
		namespace impl {
			extended::float16_t dot(int64_t n, extended::float16_t *__restrict x, int64_t incX,
									extended::float16_t *__restrict y, int64_t incY,
									cublasHandle_t *handles = &(memory::cublasHandles[0]));

			float dot(int64_t n, float *__restrict x, int64_t incX, float *__restrict y,
					  int64_t incY, cublasHandle_t *handles = &(memory::cublasHandles[0]));

			double dot(int64_t n, double *__restrict x, int64_t incX, double *__restrict y,
					   int64_t incY, cublasHandle_t *handles = &(memory::cublasHandles[0]));

			void gemv(bool trans, int64_t m, int64_t n, float alpha, const float *__restrict a,
					  int64_t lda, const float *__restrict x, int64_t incX, float beta,
					  float *__restrict y, int64_t incY,
					  cublasHandle_t *handles = &(memory::cublasHandles[0]));

			void gemv(bool trans, int64_t m, int64_t n, double alpha, const double *__restrict a,
					  int64_t lda, const double *__restrict x, int64_t incX, double beta,
					  double *__restrict y, int64_t incY,
					  cublasHandle_t *handles = &(memory::cublasHandles[0]));
		} // namespace impl

		template<>
		LR_INLINE extended::float16_t dot<device::GPU, extended::float16_t, extended::float16_t>(
		  int64_t n, extended::float16_t *__restrict x, int64_t incX,
		  extended::float16_t *__restrict y, int64_t incY) {
			return impl::dot(n, x, incX, y, incY);
		}

		template<>
		LR_INLINE float dot<device::GPU, float, float>(int64_t n, float *__restrict x, int64_t incX,
													   float *__restrict y, int64_t incY) {
			return impl::dot(n, x, incX, y, incY);
		}

		template<>
		LR_INLINE double dot<device::GPU, double, double>(int64_t n, double *__restrict x,
														  int64_t incX, double *__restrict y,
														  int64_t incY) {
			return impl::dot(n, x, incX, y, incY);
		}

		template<>
		LR_INLINE void gemv<device::GPU, float, float, float>(
		  bool trans, int64_t m, int64_t n, float alpha, const float *__restrict a, int64_t lda,
		  const float *__restrict x, int64_t incX, float beta, float *__restrict y, int64_t incY) {
			return impl::gemv(trans, m, n, alpha, a, lda, x, incX, beta, y, incY);
		}

		template<>
		LR_INLINE void
		gemv<device::GPU, double, double, double>(bool trans, int64_t m, int64_t n, double alpha,
												  const double *__restrict a, int64_t lda,
												  const double *__restrict x, int64_t incX,
												  double beta, double *__restrict y, int64_t incY) {
			return impl::gemv(trans, m, n, alpha, a, lda, x, incX, beta, y, incY);
		}
#endif
	} // namespace blas
} // namespace librapid
