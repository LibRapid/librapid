/*
 *   Copyright (c) 2009, Michael Lehn
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

#ifndef CXXBLAS_LEVEL3_HER2K_TCC
#define CXXBLAS_LEVEL3_HER2K_TCC 1

#include "cxxblas/cxxblas.h"

namespace cxxblas {

	template<typename IndexType, typename ALPHA, typename MA, typename MB, typename BETA,
			 typename MC>
	void her2k_generic(StorageOrder order, StorageUpLo upLoC, Transpose transAB, IndexType n,
					   IndexType k, const ALPHA &alpha, const MA *A, IndexType ldA, const MB *B,
					   IndexType ldB, const BETA &beta, MC *C, IndexType ldC) {
		if (order == ColMajor) {
			upLoC	= (upLoC == Upper) ? Lower : Upper;
			transAB = Transpose(transAB ^ ConjTrans);
			her2k_generic(
			  RowMajor, upLoC, transAB, n, k, conjugate(alpha), A, ldA, B, ldB, beta, C, ldC);
			return;
		}
		if ((n == 0) || (((alpha == ALPHA(0)) || (k == 0)) && (beta == BETA(1)))) { return; }
		hescal(order, upLoC, n, beta, C, ldC);
		if (alpha == ALPHA(0)) { return; }
		if (transAB == NoTrans) {
			for (IndexType l = 0; l < k; ++l) {
				her2(order, upLoC, n, alpha, A + l, ldA, B + l, ldB, C, ldC);
			}
		}
		if (transAB == Conj) { assert(0); }
		if (transAB == Trans) { assert(0); }
		if (transAB == ConjTrans) {
			for (IndexType l = 0; l < k; ++l) {
				her2(order,
					 upLoC,
					 Conj,
					 n,
					 conjugate(alpha),
					 A + l * ldA,
					 IndexType(1),
					 B + l * ldB,
					 IndexType(1),
					 C,
					 ldC);
			}
		}
	}

	template<typename IndexType, typename ALPHA, typename MA, typename MB, typename BETA,
			 typename MC>
	void her2k(StorageOrder order, StorageUpLo upLo, Transpose trans, IndexType n, IndexType k,
			   const ALPHA &alpha, const MA *A, IndexType ldA, const MB *B, IndexType ldB,
			   const BETA &beta, MC *C, IndexType ldC) {
		CXXBLAS_DEBUG_OUT("her2k_generic");

		her2k_generic(order, upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
	}

#ifdef HAVE_CBLAS

	// cher2k
	template<typename IndexType>
	typename If<IndexType>::isBlasCompatibleInteger
	her2k(StorageOrder order, StorageUpLo upLo, Transpose trans, IndexType n, IndexType k,
		  const ComplexFloat &alpha, const ComplexFloat *A, IndexType ldA, const ComplexFloat *B,
		  IndexType ldB, float beta, ComplexFloat *C, IndexType ldC) {
		CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_cher2k");

		cblas_cher2k(CBLAS::getCblasType(order),
					 CBLAS::getCblasType(upLo),
					 CBLAS::getCblasType(trans),
					 n,
					 k,
					 reinterpret_cast<const float *>(&alpha),
					 reinterpret_cast<const float *>(A),
					 ldA,
					 reinterpret_cast<const float *>(B),
					 ldB,
					 beta,
					 reinterpret_cast<float *>(C),
					 ldC);
	}

	// zher2k
	template<typename IndexType>
	typename If<IndexType>::isBlasCompatibleInteger
	her2k(StorageOrder order, StorageUpLo upLo, Transpose trans, IndexType n, IndexType k,
		  const ComplexDouble &alpha, const ComplexDouble *A, IndexType ldA, const ComplexDouble *B,
		  IndexType ldB, double beta, ComplexDouble *C, IndexType ldC) {
		CXXBLAS_DEBUG_OUT("[" BLAS_IMPL "] cblas_zher2k");

		cblas_zher2k(CBLAS::getCblasType(order),
					 CBLAS::getCblasType(upLo),
					 CBLAS::getCblasType(trans),
					 n,
					 k,
					 reinterpret_cast<const double *>(&alpha),
					 reinterpret_cast<const double *>(A),
					 ldA,
					 reinterpret_cast<const double *>(B),
					 ldB,
					 beta,
					 reinterpret_cast<double *>(C),
					 ldC);
	}

#endif // HAVE_CBLAS

} // namespace cxxblas

#endif // CXXBLAS_LEVEL3_HER2K_TCC
