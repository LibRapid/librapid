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
		ND_INLINE T cblas_dot(nd_int n, T *__restrict x, nd_int incx,
							  T *__restrict y, nd_int incy)
		{
			T res = 0;

			for (nd_int i = 0; i < n; i++)
				res += x[i * incx] * y[i * incy];

			return res;
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		ND_INLINE float cblas_dot(nd_int n, float *__restrict x, nd_int incx,
								  float *__restrict y, nd_int incy)
		{
			return cblas_sdot(n, x, incx, y, incy);
		}

		template<>
		ND_INLINE double cblas_dot(nd_int n, double *__restrict x, nd_int incx,
								   double *__restrict y, nd_int incy)
		{
			return cblas_ddot(n, x, incx, y, incy);
		}
	#endif // LIBRAPID_CBLAS

		template<typename T>
		ND_INLINE void cblas_gemv(char order, bool trans, nd_int m, nd_int n,
								  T alpha, T *__restrict a, nd_int lda,
								  T *__restrict x, nd_int incx, double beta,
								  T *__restrict y, nd_int incy)
		{
			nd_int _lda = trans ? 1 : lda;
			nd_int _fda = trans ? lda : 1;

			nd_int index_a = 0;
			nd_int index_x = 0;
			nd_int index_y = 0;

			for (nd_int outer = 0; outer < m; outer++)
			{
				if (beta == 0)
					y[index_y] = 0;
				else
					y[index_y] += y[index_y] * beta;

				index_x = 0;
				for (nd_int inner = 0; inner < n; inner++)
				{
					index_a = outer * _lda + inner * _fda;

					y[index_y] += a[index_a] * x[index_x];

					index_x = index_x + incx;
				}

				index_y++;
			}
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		ND_INLINE void cblas_gemv(char order, bool trans, nd_int m, nd_int n,
								  float alpha, float *__restrict a, nd_int lda,
								  float *__restrict x, nd_int incx, double beta,
								  float *__restrict y, nd_int incy)
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
		ND_INLINE void cblas_gemv(char order, bool trans, nd_int m, nd_int n,
								  double alpha, double *__restrict a, nd_int lda,
								  double *__restrict x, nd_int incx, double beta,
								  double *__restrict y, nd_int incy)
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
		ND_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, nd_int m,
								  nd_int n, nd_int k, T alpha,
								  T *__restrict a, nd_int lda, T *__restrict b,
								  nd_int ldb, T beta, T *__restrict c, nd_int ldc)
		{
			nd_int _lda = trans_a ? 1 : lda;
			nd_int _fda = trans_a ? lda : 1;

			nd_int _ldb = trans_a ? 1 : ldb;
			nd_int _fdb = trans_a ? ldb : 1;

			nd_int index_a, index_b, index_c;

			// Only run in parallel if arrays are smaller than a
			// given size.Running in parallel on smaller matrices
			// will result in slower code.Note, a branch is used
			// in preference to #pragma omp ... if (...) because
			// that requires runtime evaluation of a condition to
			// set up threads, which adds a significant overhead
			if (m * n * k < 25000)
			{
				for (nd_int outer = 0; outer < m; ++outer)
				{
					for (nd_int inner = 0; inner < n; ++inner)
					{
						index_c = outer * ldc + inner;

						if (beta != 0)
							c[index_c] += c[index_c] * beta;
						else
							c[index_c] = 0;

						for (nd_int sub = 0; sub < k; sub++)
						{
							index_a = outer * _lda + sub * _fda;
							index_b = inner * _ldb + sub * _fdb;

							c[index_c] += a[index_a] * b[index_b];
						}
					}
				}
			}
			else
			{
			#pragma omp parallel for shared(a, b, c, m, n, k, _lda, _ldb, ldc, _fda, _fdb, beta) private(index_a, index_b, index_c) default(none) num_threads(ND_NUM_THREADS)
				for (nd_int outer = 0; outer < m; ++outer)
				{
					for (nd_int inner = 0; inner < n; ++inner)
					{
						index_c = outer * ldc + inner;

						if (beta != 0)
							c[index_c] += c[index_c] * beta;
						else
							c[index_c] = 0;

						for (nd_int sub = 0; sub < k; sub++)
						{
							index_a = outer * _lda + sub * _fda;
							index_b = inner * _ldb + sub * _fdb;

							c[index_c] += a[index_a] * b[index_b];
						}
					}
				}
			}
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		ND_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, nd_int m,
								  nd_int n, nd_int k, float alpha,
								  float *__restrict a, nd_int lda, float *__restrict b,
								  nd_int ldb, float beta, float *__restrict c, nd_int ldc)
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
		ND_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, nd_int m,
								  nd_int n, nd_int k, double alpha,
								  double *__restrict a, nd_int lda, double *__restrict b,
								  nd_int ldb, double beta, double *__restrict c, nd_int ldc)
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