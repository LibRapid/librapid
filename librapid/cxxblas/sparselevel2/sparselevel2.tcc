/*
 *   Copyright (c) 2012, Michael Lehn
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

#ifndef CXXBLAS_SPARSELEVEL2_SPARSELEVEL2_TCC
#define CXXBLAS_SPARSELEVEL2_SPARSELEVEL2_TCC 1

namespace cxxblas {

	template<typename IndexType>
	char getIndexBaseChar(IndexType x) {
		if (x == IndexType(0)) {
			// zero-based Indexing
			return 'C';
		} else if (x == IndexType(1)) {
			// one-based Indexing
			return 'F';
		}

		// Error
		return 'E';
	}

} // namespace cxxblas

#include "cxxblas/sparselevel2/gecrsmv.tcc"
#include "cxxblas/sparselevel2/heccsmv.tcc"
#include "cxxblas/sparselevel2/hecrsmv.tcc"
#include "cxxblas/sparselevel2/syccsmv.tcc"
#include "cxxblas/sparselevel2/sycrsmv.tcc"
#include "cxxblas/sparselevel2/trccssv.tcc"
#include "cxxblas/sparselevel2/trcrssv.tcc"

#endif // CXXBLAS_SPARSELEVEL2_SPARSELEVEL2_TCC
