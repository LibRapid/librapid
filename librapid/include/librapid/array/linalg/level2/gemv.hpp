#ifndef LIBRAPID_ARRAY_LINALG_LEVEL2_GEMV_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL2_GEMV_HPP

namespace librapid::linalg {
	// trans, m, n, alpha, a, lda, x, incx, beta, y, incy

	template<typename Int, typename Alpha, typename A, typename X, typename Beta, typename Y>
	void gemv(bool trans, Int m, Int n, Alpha alpha, A *a, Int lda, X *x, Int incX, Beta beta, Y *y,
			  Int incY, backend::CPU backend = backend::CPU()) {
		cxxblas::gemv(cxxblas::StorageOrder::RowMajor,
					  (trans ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
					  static_cast<int32_t>(m),
					  static_cast<int32_t>(n),
					  alpha,
					  a,
					  static_cast<int32_t>(lda),
					  x,
					  static_cast<int32_t>(incX),
					  beta,
					  y,
					  static_cast<int32_t>(incY));
	}

#if defined(LIBRAPID_HAS_OPENCL)

	template<typename Int, typename Alpha, typename Beta>
	void gemv(bool trans, Int m, Int n, Alpha alpha, cl::Buffer a, Int lda, cl::Buffer x, Int incX,
			  Beta beta, cl::Buffer y, Int incY, backend::OpenCL) {
		using GemvScalar = decltype(alpha * beta);

		if constexpr (typetraits::IsBlasType<GemvScalar>::value) {
			auto status =
			  clblast::Gemv(clblast::Layout::kRowMajor,
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

			LIBRAPID_ASSERT(status == clblast::StatusCode::kSuccess,
							"clblast::Gemv failed: {}",
							opencl::getCLBlastErrorString(status));
		} else {
			std::string kernelNameFull =
			  std::string("gemv_") + typetraits::TypeInfo<GemvScalar>::name;
			cl::Kernel kernel(global::openCLProgram, kernelNameFull.c_str());
			kernel.setArg(0, (int)trans);
			kernel.setArg(1, static_cast<int32_t>(m));
			kernel.setArg(2, static_cast<int32_t>(n));
			kernel.setArg(3, static_cast<GemvScalar>(alpha));
			kernel.setArg(4, a);
			kernel.setArg(5, static_cast<int32_t>(lda));
			kernel.setArg(6, x);
			kernel.setArg(7, static_cast<int32_t>(incX));
			kernel.setArg(8, static_cast<GemvScalar>(beta));
			kernel.setArg(9, y);
			kernel.setArg(10, static_cast<int32_t>(incY));

			cl::NDRange globalWorkSize = cl::NDRange(m * n);

			auto status = global::openCLQueue.enqueueNDRangeKernel(
			  kernel, cl::NullRange, globalWorkSize, cl::NullRange);

			LIBRAPID_ASSERT(status == CL_SUCCESS,
							"cl::CommandQueue::enqueueNDRangeKernel GEMV call failed: {}",
							opencl::getOpenCLErrorString(status));
		}
	}

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

	template<typename Int, typename Alpha, typename Beta>
	void gemv(bool trans, Int m, Int n, Alpha alpha, float *a, Int lda, float *x, Int incX,
			  Beta beta, float *y, Int incY, backend::CUDA) {
		cublasSafeCall(cublasSgemv(global::cublasHandle,
								   (trans ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   &alpha,
								   a,
								   lda,
								   x,
								   incX,
								   &beta,
								   y,
								   incY));
	}

	template<typename Int, typename Alpha, typename Beta>
	void gemv(bool trans, Int m, Int n, Alpha alpha, double *a, Int lda, double *x, Int incX,
			  Beta beta, double *y, Int incY, backend::CUDA) {
		cublasSafeCall(cublasDgemv(global::cublasHandle,
								   (trans ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   &alpha,
								   a,
								   lda,
								   x,
								   incX,
								   &beta,
								   y,
								   incY));
	}

	template<typename Int, typename Alpha, typename Beta>
	void gemv(bool trans, Int m, Int n, Alpha alpha, Complex<float> *a, Int lda, Complex<float> *x,
			  Int incX, Beta beta, Complex<float> *y, Int incY, backend::CUDA) {
		cublasSafeCall(cublasCgemv(global::cublasHandle,
								   (trans ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   &alpha,
								   reinterpret_cast<cuComplex *>(a),
								   lda,
								   reinterpret_cast<cuComplex *>(x),
								   incX,
								   &beta,
								   reinterpret_cast<cuComplex *>(y),
								   incY));
	}

	template<typename Int, typename Alpha, typename Beta>
	void gemv(bool trans, Int m, Int n, Alpha alpha, Complex<double> *a, Int lda,
			  Complex<double> *x, Int incX, Beta beta, Complex<double> *y, Int incY,
			  backend::CUDA) {
		cublasSafeCall(cublasZgemv(global::cublasHandle,
								   (trans ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   &alpha,
								   reinterpret_cast<cuDoubleComplex *>(a),
								   lda,
								   reinterpret_cast<cuDoubleComplex *>(x),
								   incX,
								   &beta,
								   reinterpret_cast<cuDoubleComplex *>(y),
								   incY));
	}

	template<typename Int, typename Alpha, typename A, typename X, typename Beta, typename Y>
	void gemv(bool trans, Int m, Int n, Alpha alpha, A *a, Int lda, X *x, Int incX, Beta beta, Y *y,
			  Int incY, backend::CUDA) {
		jitify::Program program = global::jitCache.program(cuda::loadKernel(
		  fmt::format("{}/include/librapid/array/linalg/level2/gemv", LIBRAPID_SOURCE), false));

		Int elements = m * n;
		Int threadsPerBlock, blocksPerGrid;

		// Use 1 to 512 threads per block
		if (elements < 512) {
			threadsPerBlock = static_cast<unsigned int>(elements);
			blocksPerGrid	= 1;
		} else {
			threadsPerBlock = 512;
			blocksPerGrid	= static_cast<unsigned int>(
				ceil(static_cast<double>(elements) / static_cast<double>(threadsPerBlock)));
		}

		dim3 grid(blocksPerGrid);
		dim3 block(threadsPerBlock);

		jitifyCall(program.kernel("gemv")
					 .instantiate(jitify::reflection::Type<Int>(),
								  jitify::reflection::Type<Alpha>(),
								  jitify::reflection::Type<A>(),
								  jitify::reflection::Type<X>(),
								  jitify::reflection::Type<Beta>(),
								  jitify::reflection::Type<Y>())
					 .configure(grid, block, 0, global::cudaStream)
					 .launch(trans, m, n, alpha, a, lda, x, incX, beta, y, incY));
	}

#endif // LIBRAPID_HAS_CUDA

} // namespace librapid::linalg

#endif // LIBRAPID_ARRAY_LINALG_LEVEL2_GEMV_HPP
