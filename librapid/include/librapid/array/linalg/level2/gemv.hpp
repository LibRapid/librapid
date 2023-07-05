#ifndef LIBRAPID_ARRAY_LINALG_LEVEL2_GEMV_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL2_GEMV_HPP

namespace librapid::linalg {
	// trans, m, n, alpha, a, lda, x, incx, beta, y, incy

	template<typename Int, typename Alpha, typename A, typename X, typename Beta, typename Y>
	void gemv(bool trans, Int m, Int n, Alpha alpha, A *a, Int lda, X *x, Int incX, Beta beta, Y *y,
			  Int incY, backend::CPU backend = backend::CPU()) {
		cxxblas::gemv(cxxblas::StorageOrder::RowMajor,
					  (trans ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
					  m,
					  n,
					  alpha,
					  a,
					  lda,
					  x,
					  incX,
					  beta,
					  y,
					  incY);
	}

#if defined(LIBRAPID_HAS_OPENCL)

	template<typename Int, typename Alpha, typename Beta>
	void gemv(bool trans, Int m, Int n, Alpha alpha, cl::Buffer a, Int lda, cl::Buffer x, Int incX,
			  Beta beta, cl::Buffer y, Int incY, backend::OpenCL) {
		auto status = clblast::Gemv(clblast::Layout::kRowMajor,
									(trans ? clblast::Transpose::kYes : clblast::Transpose::kNo),
									m,
									n,
									alpha,
									a(),
									0,
									lda,
									x(),
									0,
									incX,
									beta,
									y(),
									0,
									incY,
									&global::openCLQueue());
	}

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

#endif // LIBRAPID_HAS_CUDA

} // namespace librapid::linalg

#endif // LIBRAPID_ARRAY_LINALG_LEVEL2_GEMV_HPP
