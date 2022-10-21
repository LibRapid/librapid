/*
 *   Copyright (c) 2010, Michael Lehn
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CXXBLAS_LEVEL2_HPMV_TCC
#define CXXBLAS_LEVEL2_HPMV_TCC 1

#include <complex>
#include "cxxblas/cxxblas.h"

namespace cxxblas {

	template<typename IndexType, typename ALPHA, typename MA, typename VX, typename BETA,
			 typename VY>
	void hpmv_generic(StorageOrder order, StorageUpLo upLo, Transpose conjugateA, IndexType n,
					  const ALPHA &alpha, const MA *A, const VX *x, IndexType incX,
					  const BETA &beta, VY *y, IndexType incY) {
		if (order == ColMajor) {
			upLo	   = (upLo == Upper) ? Lower : Upper;
			conjugateA = Transpose(conjugateA ^ Conj);
		}
		scal_init_generic(n, beta, y, incY);
		if (upLo == Upper) {
			if (conjugateA == Conj) {
				for (IndexType i = 0, iY = 0, iX = 0; i < n; ++i, iX += incX, iY += incY) {
					y[iY] += alpha * cxxblas::real(A[i * (2 * n - i + 1) / 2]) * x[iX];

					VY y_ = VY(0);
					dot_generic(n - i - 1,
								A + i * (2 * n - i + 1) / 2 + 1,
								IndexType(1),
								x + iX + incX,
								incX,
								y_);
					y[iY] += alpha * y_;
					axpy_generic(n - i - 1,
								 alpha * x[iX],
								 A + i * (2 * n - i + 1) / 2 + 1,
								 IndexType(1),
								 y + iY + incY,
								 incY);
				}
			} else {
				for (IndexType i = 0, iY = 0, iX = 0; i < n; ++i, iX += incX, iY += incY) {
					y[iY] += alpha * cxxblas::real(A[i * (2 * n - i + 1) / 2]) * x[iX];

					VY y_ = VY(0);
					dotu_generic(n - i - 1,
								 A + i * (2 * n - i + 1) / 2 + 1,
								 IndexType(1),
								 x + iX + incX,
								 incX,
								 y_);
					y[iY] += alpha * y_;
					acxpy_generic(n - i - 1,
								  alpha * x[iX],
								  A + i * (2 * n - i + 1) / 2 + 1,
								  IndexType(1),
								  y + iY + incY,
								  incY);
				}
			}
		} else {
			if (conjugateA == Conj) {
				for (IndexType i = 0, iY = 0, iX = 0; i < n; ++i, iX += incX, iY += incY) {
					y[iY] += alpha * cxxblas::real(A[i + i * (i + 1) / 2]) * x[iX];

					VY y_ = VY(0);
					dot_generic(i, A + i * (i + 1) / 2, IndexType(1), x, incX, y_);
					y[iY] += alpha * y_;
					axpy_generic(i, alpha * x[iX], A + i * (i + 1) / 2, IndexType(1), y, incY);
				}
			} else {
				for (IndexType i = 0, iY = 0, iX = 0; i < n; ++i, iX += incX, iY += incY) {
					y[iY] += alpha * cxxblas::real(A[i + i * (i + 1) / 2]) * x[iX];

					VY y_ = VY(0);
					dotu_generic(i, A + i * (i + 1) / 2, IndexType(1), x, incX, y_);
					y[iY] += alpha * y_;
					acxpy_generic(i, alpha * x[iX], A + i * (i + 1) / 2, IndexType(1), y, incY);
				}
			}
		}
	}

	//------------------------------------------------------------------------------

	template<typename IndexType, typename ALPHA, typename MA, typename VX, typename BETA,
			 typename VY>
	void hpmv(StorageOrder order, StorageUpLo upLo, IndexType n, const ALPHA &alpha, const MA *A,
			  const VX *x, IndexType incX, const BETA &beta, VY *y, IndexType incY) {
		CXXBLAS_DEBUG_OUT("hpmv_generic");

		if (incX < 0) { x -= incX * (n - 1); }
		if (incY < 0) { y -= incY * (n - 1); }
		hpmv_generic(order, upLo, NoTrans, n, alpha, A, x, incX, beta, y, incY);
	}

#ifdef HAVE_CBLAS

	// chpmv
	template<typename IndexType>
	typename If<IndexType>::isBlasCompatibleInteger
	hpmv(StorageOrder order, StorageUpLo upLo, IndexType n, const ComplexFloat &alpha,
		 const ComplexFloat *A, const ComplexFloat *x, IndexType incX, const ComplexFloat &beta,
		 ComplexFloat *y, IndexType incY) {
		CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_chpmv");

		cblas_chpmv(CBLAS::getCblasType(order),
					CBLAS::getCblasType(upLo),
					n,
					reinterpret_cast<const float *>(&alpha),
					reinterpret_cast<const float *>(A),
					reinterpret_cast<const float *>(x),
					incX,
					reinterpret_cast<const float *>(&beta),
					reinterpret_cast<float *>(y),
					incY);
	}

	// zhpmv
	template<typename IndexType>
	typename If<IndexType>::isBlasCompatibleInteger
	hpmv(StorageOrder order, StorageUpLo upLo, IndexType n, const ComplexDouble &alpha,
		 const ComplexDouble *A, const ComplexDouble *x, IndexType incX, const ComplexDouble &beta,
		 ComplexDouble *y, IndexType incY) {
		CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zhpmv");

		cblas_zhpmv(CBLAS::getCblasType(order),
					CBLAS::getCblasType(upLo),
					n,
					reinterpret_cast<const double *>(&alpha),
					reinterpret_cast<const double *>(A),
					reinterpret_cast<const double *>(x),
					incX,
					reinterpret_cast<const double *>(&beta),
					reinterpret_cast<double *>(y),
					incY);
	}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL2_HPMV_TCC
