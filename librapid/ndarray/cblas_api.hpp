#ifndef LIBRAPID_CBLAS_API
#define LIBRAPID_CBLAS_API

#include <librapid/config.hpp>

#ifdef LIBRAPID_CBLAS
#include <cblas.h>
#endif // LIBRAPID_CBLAS

namespace librapid
{
	namespace linalg
	{
		template<typename T>
		LR_INLINE T cblas_dot(lr_int n, T *__restrict x, lr_int incx,
							  T *__restrict y, lr_int incy)
		{
			T res = 0;

			for (lr_int i = 0; i < n; i++)
				res += x[i * incx] * y[i * incy];

			return res;
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		LR_INLINE float cblas_dot(lr_int n, float *__restrict x, lr_int incx,
								  float *__restrict y, lr_int incy)
		{
			return cblas_sdot(n, x, incx, y, incy);
		}

		template<>
		LR_INLINE double cblas_dot(lr_int n, double *__restrict x, lr_int incx,
								   double *__restrict y, lr_int incy)
		{
			return cblas_ddot(n, x, incx, y, incy);
		}
	#endif // LIBRAPID_CBLAS

		template<typename T>
		LR_INLINE void cblas_gemv_no_blas(char order, bool trans, lr_int m, lr_int n,
										  T alpha, T *__restrict a, lr_int lda,
										  T *__restrict x, lr_int incx, T beta,
										  T *__restrict y, lr_int incy)
		{
			lr_int _lda = trans ? 1 : lda;
			lr_int _fda = trans ? lda : 1;

			lr_int index_a = 0;
			lr_int index_x = 0;
			lr_int index_y = 0;

			for (lr_int outer = 0; outer < m; outer++)
			{
				if (beta == 0)
					y[index_y] = 0;
				else
					y[index_y] += y[index_y] * beta;

				index_x = 0;
				for (lr_int inner = 0; inner < n; inner++)
				{
					index_a = outer * _lda + inner * _fda; // A contains invalid values?

					y[index_y] += a[index_a] * x[index_x];

					index_x = index_x + incx;
				}

				index_y++;
			}
		}

		template<typename T>
		LR_INLINE void cblas_gemv(char order, bool trans, lr_int m, lr_int n, T alpha,
								  T *__restrict a, lr_int lda, T *__restrict x, lr_int incx,
								  T beta, T *__restrict y, lr_int incy)
		{
			cblas_gemv_no_blas(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		LR_INLINE void cblas_gemv(char order, bool trans, lr_int m, lr_int n,
								  float alpha, float *__restrict a, lr_int lda,
								  float *__restrict x, lr_int incx, float beta,
								  float *__restrict y, lr_int incy)
		{
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

			cblas_sgemv(blas_order, blas_trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
		}

		template<>
		LR_INLINE void cblas_gemv(char order, bool trans, lr_int m, lr_int n,
								  double alpha, double *__restrict a, lr_int lda,
								  double *__restrict x, lr_int incx, double beta,
								  double *__restrict y, lr_int incy)
		{
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

			cblas_dgemv(blas_order, blas_trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
		}
	#endif // LIBRAPID_CBLAS

		template<typename T>
		LR_INLINE void cblas_gemm_no_blas(char order, bool trans_a, bool trans_b, lr_int m,
										  lr_int n, lr_int k, T alpha,
										  T *__restrict a, lr_int lda, T *__restrict b,
										  lr_int ldb, T beta, T *__restrict c, lr_int ldc)
		{
			lr_int temp_a, index_c;

			// #pragma omp parallel for shared(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) private(temp_a, index_c) default(none)
			for (lr_int outer = 0; outer < m; outer++)
			{
				for (lr_int inner = 0; inner < n; inner++)
				{
					temp_a = outer * lda;
					index_c = inner + outer * ldc;

					c[index_c] = 0;

					for (lr_int sub = 0; sub < k; sub++)
					{
						c[index_c] += a[temp_a + sub] * b[sub * ldb + inner];
					}
				}
			}
		}

		template<typename T>
		LR_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, lr_int m,
								  lr_int n, lr_int k, T alpha,
								  T *__restrict a, lr_int lda, T *__restrict b,
								  lr_int ldb, T beta, T *__restrict c, lr_int ldc)
		{
			cblas_gemm_no_blas(order, trans_a, trans_b, m, n, k,
							   alpha, a, lda, b, ldb, beta, c, ldc);
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		LR_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, lr_int m,
								  lr_int n, lr_int k, float alpha,
								  float *__restrict a, lr_int lda, float *__restrict b,
								  lr_int ldb, float beta, float *__restrict c, lr_int ldc)
		{
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
						m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
		}

		template<>
		LR_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, lr_int m,
								  lr_int n, lr_int k, double alpha,
								  double *__restrict a, lr_int lda, double *__restrict b,
								  lr_int ldb, double beta, double *__restrict c, lr_int ldc)
		{
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

			cblas_dgemm(blas_order, blas_trans_a, blas_trans_b, m, n, k,
						alpha, a, lda, b, ldb, beta, c, ldc);
		}
	#endif // LIBRAPID_CBLAS
	}
}

#endif // LIBRAPID_CBLAS_API