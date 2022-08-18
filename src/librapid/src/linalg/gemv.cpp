#include <librapid/internal/config.hpp>
#include <librapid/linalg/blasInterface.hpp>
#include <librapid/modified/modified.hpp>

namespace librapid::blas {
#if defined(LIBRAPID_HAS_BLAS)
	template<>
	void gemv<device::CPU, float, float, float>(bool trans, int64_t m, int64_t n, float alpha,
												const float *__restrict a, int64_t lda,
												const float *__restrict x, int64_t incX, float beta,
												float *__restrict y, int64_t incY) {
		cblas_sgemv(CblasRowMajor,
					trans ? CblasTrans : CblasNoTrans,
					(blasint)m,
					(blasint)n,
					alpha,
					a,
					(blasint)lda,
					x,
					(blasint)incX,
					beta,
					y,
					(blasint)incY);
	}

	template<>
	void gemv<device::CPU, double, double, double>(bool trans, int64_t m, int64_t n, double alpha,
												   const double *__restrict a, int64_t lda,
												   const double *__restrict x, int64_t incX,
												   double beta, double *__restrict y,
												   int64_t incY) {
		cblas_dgemv(CblasRowMajor,
					trans ? CblasTrans : CblasNoTrans,
					(blasint)m,
					(blasint)n,
					alpha,
					a,
					(blasint)lda,
					x,
					(blasint)incX,
					beta,
					y,
					(blasint)incY);
	}  // namespace impl
#endif // LIBRAPID_HAS_BLAS

#if defined(LIBRAPID_HAS_CUDA)
	namespace impl {
		void gemv(bool trans, int64_t m, int64_t n, float alpha, const float *__restrict a,
				  int64_t lda, const float *__restrict x, int64_t incX, float beta,
				  float *__restrict y, int64_t incY, cublasHandle_t *handles) {
#	if defined(LIBRAPID_HAS_OMP)
			int64_t threadNum = omp_get_thread_num();
#	else
			int64_t threadNum = 0;
#	endif

			cublasSafeCall(cublasSgemv_v2(handles[threadNum],
										  trans ? CUBLAS_OP_T : CUBLAS_OP_N,
										  m,
										  n,
										  &alpha,
										  a,
										  lda,
										  x,
										  incX,
										  &beta,
										  y,
										  incY));
		}

		void gemv(bool trans, int64_t m, int64_t n, double alpha, const double *__restrict a,
				  int64_t lda, const double *__restrict x, int64_t incX, double beta,
				  double *__restrict y, int64_t incY, cublasHandle_t *handles) {
#	if defined(LIBRAPID_HAS_OMP)
			int64_t threadNum = omp_get_thread_num();
#	else
			int64_t threadNum = 0;
#	endif

			cublasSafeCall(cublasDgemv_v2(handles[threadNum],
										  trans ? CUBLAS_OP_T : CUBLAS_OP_N,
										  m,
										  n,
										  &alpha,
										  a,
										  lda,
										  x,
										  incX,
										  &beta,
										  y,
										  incY));
		}
	}  // namespace impl
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::blas