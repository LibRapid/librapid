#include <librapid/internal/config.hpp>
#include <librapid/linalg/blasInterface.hpp>
#include <librapid/internal/memUtils.hpp>
#include <librapid/modified/modified.hpp>

namespace librapid::blas {
#if defined(LIBRAPID_HAS_BLAS)
	template<>
	void gemm<device::CPU, float, float, float>(bool transA, bool transB, int64_t m, int64_t n,
												int64_t k, float alpha, const float *__restrict a,
												int64_t lda, const float *__restrict b, int64_t ldb,
												float beta, float *__restrict c, int64_t ldc) {
		CBLAS_TRANSPOSE tmpTransA = CblasNoTrans, tmpTransB = CblasNoTrans;
		if (transA) tmpTransA = CblasTrans;
		if (transB) tmpTransB = CblasTrans;
		cblas_sgemm(CblasRowMajor,
					tmpTransA,
					tmpTransB,
					(blasint)m,
					(blasint)n,
					(blasint)k,
					alpha,
					a,
					(blasint)lda,
					b,
					(blasint)ldb,
					beta,
					c,
					(blasint)ldc);
	}

	template<>
	void gemm<device::CPU, double, double, double>(bool transA, bool transB, int64_t m, int64_t n,
												   int64_t k, double alpha,
												   const double *__restrict a, int64_t lda,
												   const double *__restrict b, int64_t ldb,
												   double beta, double *__restrict c, int64_t ldc) {
		CBLAS_TRANSPOSE tmpTransA = CblasNoTrans, tmpTransB = CblasNoTrans;
		if (transA) tmpTransA = CblasTrans;
		if (transB) tmpTransB = CblasTrans;
		cblas_dgemm(CblasRowMajor,
					tmpTransA,
					tmpTransB,
					(blasint)m,
					(blasint)n,
					(blasint)k,
					alpha,
					a,
					(blasint)lda,
					b,
					(blasint)ldb,
					beta,
					c,
					(blasint)ldc);
	}
#endif
} // namespace librapid::blas
