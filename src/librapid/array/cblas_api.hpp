#ifndef LIBRAPID_HAS_BLAS_API
#define LIBRAPID_HAS_BLAS_API

#include <thread>
#include <librapid/config.hpp>

#ifdef LIBRAPID_HAS_BLAS

#include <cblas.h>

#endif // LIBRAPID_HAS_BLAS

#ifdef LIBRAPID_HAS_CUDA

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <library_types.h>

#endif // LIBRAPID_HAS_CUDA

namespace librapid::linalg {
	template<typename A, typename B>
	// using common = typename std::common_type<A, B>::type;
	using common = typename CommonType<A, B>::type;

	template<typename A, typename B>
	inline common<A, B> cblas_dot(int64_t n, A *__restrict x, int64_t incx, B *__restrict y, int64_t incy) {
		common<A, B> res = 0;
		for (int64_t i = 0; i < n; i++)
			res += x[i * incx] * y[i * incy];
		return res;
	}

#ifdef LIBRAPID_HAS_BLAS

	template<>
	inline float cblas_dot(int64_t n, float *__restrict x, int64_t incx,
						   float *__restrict y, int64_t incy) {
		return cblas_sdot((int) n, x, (int) incx, y, (int) incy);
	}

	template<>
	inline double cblas_dot(int64_t n, double *__restrict x, int64_t incx,
							double *__restrict y, int64_t incy) {
		return cblas_ddot((int) n, x, (int) incx, y, (int) incy);
	}

#endif // LIBRAPID_HAS_BLAS

	template<typename A, typename B, typename C>
	inline void cblas_gemv_no_blas(char order, bool trans, int64_t m, int64_t n,
								   A alpha, A *__restrict a, int64_t lda,
								   B *__restrict x, int64_t incx, C beta,
								   C *__restrict y, int64_t incy) {
		int64_t _lda = trans ? 1 : lda;
		int64_t _fda = trans ? lda : 1;

		int64_t index_a = 0;
		int64_t index_x = 0;
		int64_t index_y = 0;

		for (int64_t outer = 0; outer < m; outer++) {
			if (beta == 0)
				y[index_y] = 0;
			else
				y[index_y] += y[index_y] * beta;

			index_x = 0;
			for (int64_t inner = 0; inner < n; inner++) {
				index_a = outer * _lda + inner * _fda;

				y[index_y] += a[index_a] * x[index_x];

				index_x = index_x + incx;
			}

			index_y++;
		}
	}

	template<typename A, typename B, typename C>
	inline void cblas_gemv(char order, bool trans, int64_t m, int64_t n, A alpha,
						   A *__restrict a, int64_t lda, B *__restrict x, int64_t incx,
						   C beta, C *__restrict y, int64_t incy) {
		cblas_gemv_no_blas(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
	}

#ifdef LIBRAPID_HAS_BLAS

	template<>
	inline void cblas_gemv(char order, bool trans, int64_t m, int64_t n,
						   float alpha, float *__restrict a, int64_t lda,
						   float *__restrict x, int64_t incx, float beta,
						   float *__restrict y, int64_t incy) {
		CBLAS_ORDER blas_order{};
		if (order == 'r')
			blas_order = CblasRowMajor;
		else
			blas_order = CblasColMajor;

		CBLAS_TRANSPOSE blas_trans{};
		if (trans)
			blas_trans = CblasTrans;
		else
			blas_trans = CblasNoTrans;

		cblas_sgemv(blas_order, blas_trans, (int) m, (int) n, alpha, a, (int) lda, x, (int) incx, beta, y, (int) incy);
	}

	template<>
	inline void cblas_gemv(char order, bool trans, int64_t m, int64_t n,
						   double alpha, double *__restrict a, int64_t lda,
						   double *__restrict x, int64_t incx, double beta,
						   double *__restrict y, int64_t incy) {
		CBLAS_ORDER blas_order{};
		if (order == 'r')
			blas_order = CblasRowMajor;
		else
			blas_order = CblasColMajor;

		CBLAS_TRANSPOSE blas_trans{};
		if (trans)
			blas_trans = CblasTrans;
		else
			blas_trans = CblasNoTrans;

		cblas_dgemv(blas_order, blas_trans, (int) m, (int) n, alpha, a, (int) lda, x, (int) incx, beta, y, (int) incy);
	}

#endif // LIBRAPID_HAS_BLAS

	template<typename A, typename B, typename C>
	inline void cblas_gemm_no_blas(char order, bool trans_a, bool trans_b, int64_t m,
								   int64_t n, int64_t k, A alpha,
								   A *__restrict a, int64_t lda, B *__restrict b,
								   int64_t ldb, C beta, C *__restrict c, int64_t ldc) {
		int64_t outer, inner, sub;
		int64_t temp_a, index_c;

		int64_t _lda = trans_a ? 1 : lda;
		int64_t _fda = trans_a ? lda : 1;

		int64_t _ldb = trans_b ? 1 : ldb;
		int64_t _fdb = trans_b ? ldb : 1;

		int64_t _ldc = trans_b ? 1 : ldc;
		int64_t _fdc = trans_b ? ldc : 1;

		if (m * n * k < 2500) {
			for (outer = 0; outer < m; ++outer) {
				for (inner = 0; inner < n; ++inner) {
					temp_a = outer * _lda;
					index_c = inner * _fdc + outer * _ldc;

					if (beta == 0)
						c[index_c] = 0;
					else
						c[index_c] += c[index_c] * beta;

					for (sub = 0; sub < k; ++sub) {
						c[index_c] += a[temp_a + sub * _fda] * b[sub * _ldb + inner * _fdb];
					}
				}
			}
		} else {
#pragma omp parallel for shared(m, n, k, _lda, _fda, _ldb, _fdb, _ldc, _fdc, alpha, beta, a, b, c) private(outer, inner, sub, temp_a, index_c) default(none)
			for (outer = 0; outer < m; ++outer) {
				for (inner = 0; inner < n; ++inner) {
					temp_a = outer * _lda;
					index_c = inner * _fdc + outer * _ldc;

					if (beta == 0)
						c[index_c] = 0;
					else
						c[index_c] += c[index_c] * beta;

					for (sub = 0; sub < k; ++sub) {
						c[index_c] += a[temp_a + sub * _fda] * b[sub * _ldb + inner * _fdb];
					}
				}
			}
		}
	}

	template<typename A, typename B, typename C>
	inline void cblas_gemm(char order, bool trans_a, bool trans_b, int64_t m,
						   int64_t n, int64_t k, A alpha,
						   A *__restrict a, int64_t lda, B *__restrict b,
						   int64_t ldb, C beta, C *__restrict c, int64_t ldc) {
		cblas_gemm_no_blas(order, trans_a, trans_b, m, n, k,
						   alpha, a, lda, b, ldb, beta, c, ldc);
	}

#ifdef LIBRAPID_HAS_BLAS

	template<>
	inline void cblas_gemm(char order, bool trans_a, bool trans_b, int64_t m,
						   int64_t n, int64_t k, float alpha,
						   float *__restrict a, int64_t lda, float *__restrict b,
						   int64_t ldb, float beta, float *__restrict c, int64_t ldc) {
		CBLAS_ORDER blas_order{};
		if (order == 'r')
			blas_order = CblasRowMajor;
		else
			blas_order = CblasColMajor;

		CBLAS_TRANSPOSE blas_trans_a{};
		if (trans_a)
			blas_trans_a = CblasTrans;
		else
			blas_trans_a = CblasNoTrans;

		CBLAS_TRANSPOSE blas_trans_b;
		if (trans_b)
			blas_trans_b = CblasTrans;
		else
			blas_trans_b = CblasNoTrans;

		cblas_sgemm(blas_order, blas_trans_a, blas_trans_b,
					(int) m, (int) n, (int) k, alpha, a, (int) lda, b, (int) ldb, beta, c, (int) ldc);
	}

	template<>
	inline void cblas_gemm(char order, bool trans_a, bool trans_b, int64_t m,
						   int64_t n, int64_t k, double alpha,
						   double *__restrict a, int64_t lda, double *__restrict b,
						   int64_t ldb, double beta, double *__restrict c, int64_t ldc) {
		CBLAS_ORDER blas_order{};
		if (order == 'r')
			blas_order = CblasRowMajor;
		else
			blas_order = CblasColMajor;

		CBLAS_TRANSPOSE blas_trans_a{};
		if (trans_a)
			blas_trans_a = CblasTrans;
		else
			blas_trans_a = CblasNoTrans;

		CBLAS_TRANSPOSE blas_trans_b;
		if (trans_b)
			blas_trans_b = CblasTrans;
		else
			blas_trans_b = CblasNoTrans;

		cblas_dgemm(blas_order, blas_trans_a, blas_trans_b, (int) m, (int) n, (int) k,
					alpha, a, (int) lda, b, (int) ldb, beta, c, (int) ldc);
	}

#endif // LIBRAPID_HAS_BLAS

#ifdef LIBRAPID_HAS_CUDA

	template<typename A, typename B, typename C>
	inline void cblas_dot_cuda(cublasHandle_t &handle, int64_t n, A *__restrict x, int64_t incx,
									   B *__restrict y, int64_t incy, C *__restrict c) {
		// TODO: Write custom CUDA kernel for vector dot product
		*c = 0;
	}

	template<>
	inline void cblas_dot_cuda(cublasHandle_t &handle, int64_t n, float *__restrict x, int64_t incx,
								float *__restrict y, int64_t incy, float *__restrict c) {
		cublasSdot_v2(handle, (int) n, x, (int) incx, y, (int) incy, c);
	}

	template<>
	inline void cblas_dot_cuda(cublasHandle_t &handle, int64_t n, double *__restrict x, int64_t incx,
								 double *__restrict y, int64_t incy, double *__restrict c) {
		cublasDdot_v2(handle, (int) n, x, (int) incx, y, (int) incy, c);
	}

	template<typename A, typename B, typename C>
	inline void cblas_gemm_cuda(cublasHandle_t &handle, bool trans_a, bool trans_b, int64_t m,
								int64_t n, int64_t k, A alpha,
								A *__restrict a, int64_t lda, B *__restrict b,
								int64_t ldb, C beta, C *__restrict c, int64_t ldc) {
		cublasOperation_t blas_trans_a{};
		if (trans_a)
			blas_trans_a = CUBLAS_OP_T;
		else
			blas_trans_a = CUBLAS_OP_N;

		cublasOperation_t blas_trans_b;
		if (trans_b)
			blas_trans_b = CUBLAS_OP_T;
		else
			blas_trans_b = CUBLAS_OP_N;

		// TODO: Create CUDA kernel for matrix multiplication with arbitrary datatypes
	}

	template<>
	inline void cblas_gemm_cuda(cublasHandle_t &handle, bool trans_a, bool trans_b, int64_t m,
								int64_t n, int64_t k, float alpha,
								float *__restrict a, int64_t lda, float *__restrict b,
								int64_t ldb, float beta, float *__restrict c, int64_t ldc) {
		cublasOperation_t blas_trans_a{};
		if (trans_a)
			blas_trans_a = CUBLAS_OP_T;
		else
			blas_trans_a = CUBLAS_OP_N;

		cublasOperation_t blas_trans_b;
		if (trans_b)
			blas_trans_b = CUBLAS_OP_T;
		else
			blas_trans_b = CUBLAS_OP_N;

		cublasSafeCall(cublasSgemm_v2(handle, blas_trans_a, blas_trans_b,
									  (int) n, (int) m, (int) k, &alpha, b, (int) ldb, a, (int) lda, &beta, c,
									  (int) ldc));
	}

	template<>
	inline void cblas_gemm_cuda(cublasHandle_t &handle, bool trans_a, bool trans_b, int64_t m,
								int64_t n, int64_t k, double alpha,
								double *__restrict a, int64_t lda, double *__restrict b,
								int64_t ldb, double beta, double *__restrict c, int64_t ldc) {
		cublasOperation_t blas_trans_a{};
		if (trans_a)
			blas_trans_a = CUBLAS_OP_T;
		else
			blas_trans_a = CUBLAS_OP_N;

		cublasOperation_t blas_trans_b;
		if (trans_b)
			blas_trans_b = CUBLAS_OP_T;
		else
			blas_trans_b = CUBLAS_OP_N;

		// ublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,d_b,n,d_a,k,&bet,d_c,n)
		cublasSafeCall(cublasDgemm_v2(handle, blas_trans_a, blas_trans_b,
									  (int) n, (int) m, (int) k, &alpha, b, (int) ldb, a, (int) lda, &beta, c,
									  (int) ldc));
	}

#endif // LIBRAPID_HAS_CUDA
}

#endif // LIBRAPID_HAS_BLAS_API