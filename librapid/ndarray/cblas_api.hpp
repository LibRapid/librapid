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
		ND_INLINE T cblas_dot(nd_int n, const T *__restrict x, const nd_int incx,
							  const T *__restrict y, const nd_int incy)
		{
			T res = 0;

			for (nd_int i = 0; i < n; i++)
				res += x[i * incx] * y[i * incy];

			return res;
		}

	#ifdef LIBRAPID_CBLAS
		template<>
		ND_INLINE float cblas_dot(nd_int n, const float *__restrict x, const nd_int incx,
								  const float *__restrict y, const nd_int incy)
		{
			return cblas_sdot(n, x, incx, y, incy);
		}

		template<>
		ND_INLINE double cblas_dot(nd_int n, const double *__restrict x, const nd_int incx,
								   const double *__restrict y, const nd_int incy)
		{
			return cblas_ddot(n, x, incx, y, incy);
		}
	#endif // LIBRAPID_CBLAS

		template<typename T>
		ND_INLINE void cblas_gemv(char order, bool trans, const nd_int m, const nd_int n,
								  const T alpha, const T *__restrict a, const nd_int lda,
								  const T *__restrict x, const nd_int incx, const double beta,
								  T *__restrict y, const nd_int incy)
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
		ND_INLINE void cblas_gemv(char order, bool trans, const nd_int m, const nd_int n,
								  const float alpha, const float *__restrict a, const nd_int lda,
								  const float *__restrict x, const nd_int incx, const double beta,
								  float *__restrict y, const nd_int incy)
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
		ND_INLINE void cblas_gemv(char order, bool trans, const nd_int m, const nd_int n,
								  const double alpha, const double *__restrict a, const nd_int lda,
								  const double *__restrict x, const nd_int incx, const double beta,
								  double *__restrict y, const nd_int incy)
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
		ND_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, const nd_int m,
								  const nd_int n, const nd_int k, const T alpha,
								  const T *__restrict a, const nd_int lda, const T *__restrict b,
								  const nd_int ldb, const T beta, T *__restrict c, nd_int ldc)
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
					// for (nd_int inner = 0; inner < K_prime; ++inner)
					for (nd_int inner = 0; inner < n; ++inner)
					{
						index_c = outer * ldc + inner;
						c[index_c] = 0;

						for (nd_int sub = 0; sub < k; sub++)
						{
							index_a = outer * _lda + sub * _fda;
							index_b = inner * _ldb + sub * _fdb;

							if (beta != 0)
								c[index_c] += c[index_c] * beta;

							c[index_c] += a[index_a] * b[index_b];
						}
					}
				}
			}
			else
			{
			#pragma omp parallel for shared(a, b, c, m, n, k, _lda, _ldb, _fda, _fdb) private(index_a, index_b, index_c) default(none) num_threads(ND_NUM_THREADS)
				for (nd_int outer = 0; outer < m; ++outer)
				{
					// for (nd_int inner = 0; inner < K_prime; ++inner)
					for (nd_int inner = 0; inner < n; ++inner)
					{
						index_c = outer * ldc + inner;
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
		ND_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, const nd_int m,
								  const nd_int n, const nd_int k, const float alpha,
								  const float *__restrict a, const nd_int lda, const float *__restrict b,
								  const nd_int ldb, const float beta, float *__restrict c, nd_int ldc)
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
		ND_INLINE void cblas_gemm(char order, bool trans_a, bool trans_b, const nd_int m,
								  const nd_int n, const nd_int k, const double alpha,
								  const double *__restrict a, const nd_int lda, const double *__restrict b,
								  const nd_int ldb, const double beta, double *__restrict c, nd_int ldc)
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