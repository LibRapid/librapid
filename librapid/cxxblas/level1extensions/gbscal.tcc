/*
 *   Copyright (c) 2012, Michael Lehn, Klaus Pototzky
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

#ifndef CXXBLAS_LEVEL1EXTENSIONS_GBSCAL_TCC
#define CXXBLAS_LEVEL1EXTENSIONS_GBSCAL_TCC 1

#include <cassert>
#include "cxxblas/cxxblas.h"

namespace cxxblas {

	template<typename IndexType, typename ALPHA, typename MA>
	void gbscal(StorageOrder order, IndexType m, IndexType n, IndexType kl, IndexType ku,
				const ALPHA &alpha, MA *A, IndexType ldA) {
		CXXBLAS_DEBUG_OUT("gbscal_generic");

		if (order == ColMajor) {
			std::swap(m, n);
			std::swap(kl, ku);
		}
		using std::max;
		using std::min;

		for (IndexType j = 0, i = -kl; i <= ku; ++j, ++i) {
			IndexType length = (i < 0) ? min(m + i, min(m, n)) : min(n - i, min(m, n));
			scal(length, alpha, A + j + max(-i, 0) * ldA, ldA);
		}
		return;
	}

} // namespace cxxblas

#endif // CXXBLAS_LEVEL1EXTENSIONS_GBSCAL_TCC
