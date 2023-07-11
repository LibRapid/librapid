#ifndef LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP

namespace librapid::linalg {
	template<typename Int, typename Alpha, typename A, typename B, typename Beta, typename C>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, A *a, Int lda, B *b,
			  Int ldb, Beta beta, C *c, Int ldc, backend::CPU backend = backend::CPU()) {
		cxxblas::gemm(cxxblas::StorageOrder::RowMajor,
					  (transA ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
					  (transB ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
					  m,
					  n,
					  k,
					  alpha,
					  a,
					  lda,
					  b,
					  ldb,
					  beta,
					  c,
					  ldc);
	}

#if defined(LIBRAPID_HAS_OPENCL)

	template<typename Int, typename Alpha, typename Beta>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, cl::Buffer a, Int lda,
			  cl::Buffer b, Int ldb, Beta beta, cl::Buffer c, Int ldc, backend::OpenCL) {
		using GemmScalar = decltype(alpha * beta);

		if constexpr (typetraits::IsBlasType<GemmScalar>::value) {
			auto status = clblast::Gemm<GemmScalar>(
			  clblast::Layout::kRowMajor,
			  (transA ? clblast::Transpose::kYes : clblast::Transpose::kNo),
			  (transB ? clblast::Transpose::kYes : clblast::Transpose::kNo),
			  m,
			  n,
			  k,
			  alpha,
			  a(),
			  0,
			  lda,
			  b(),
			  0,
			  ldb,
			  beta,
			  c(),
			  0,
			  ldc,
			  &global::openCLQueue());

			LIBRAPID_ASSERT(status == clblast::StatusCode::kSuccess,
							"clblast::Gemm failed: {}",
							opencl::getCLBlastErrorString(status));
		} else {
			std::string kernelNameFull =
			  std::string("gemm_") + typetraits::TypeInfo<GemmScalar>::name;
			cl::Kernel kernel(global::openCLProgram, kernelNameFull.c_str());
			kernel.setArg(0, (int)transA);
			kernel.setArg(1, (int)transB);
			kernel.setArg(2, (int32_t)m);
			kernel.setArg(3, (int32_t)n);
			kernel.setArg(4, (int32_t)k);
			kernel.setArg(5, (GemmScalar)alpha);
			kernel.setArg(6, a);
			kernel.setArg(7, (int32_t)lda);
			kernel.setArg(8, b);
			kernel.setArg(9, (int32_t)ldb);
			kernel.setArg(10, (GemmScalar)beta);
			kernel.setArg(11, c);
			kernel.setArg(12, (int32_t)ldc);

			size_t TS = 32; // Must be the same as in the kernel (line 1 of gemm.cu)

			cl::NDRange globalWorkSize =
			  cl::NDRange(((n - 1) / TS + 1) * TS, ((m - 1) / TS + 1) * TS);
			cl::NDRange localWorkSize = cl::NDRange(TS, TS);

			auto status = global::openCLQueue.enqueueNDRangeKernel(
			  kernel, cl::NullRange, globalWorkSize, localWorkSize);

			LIBRAPID_ASSERT(status == CL_SUCCESS,
							"cl::CommandQueue::enqueueNDRangeKernel GEMM call failed: {}",
							opencl::getOpenCLErrorString(status));
		}
	}

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

	struct CuBLASGemmComputeType {
		cublasComputeType_t computeType;
		cublasDataType_t scaleType;
	};

	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE CuBLASGemmComputeType
	cublasGemmComputeType(cublasDataType_t a, cublasDataType_t b, cublasDataType_t c) {
#	if defined(LIBRAPID_FAST_MATH)
		constexpr auto selector = [](CuBLASGemmComputeType fast, CuBLASGemmComputeType) {
			return fast;
		};
#	else
		constexpr auto selector = [](CuBLASGemmComputeType, CuBLASGemmComputeType precise) {
			return precise;
		};
#	endif

		LIBRAPID_ASSERT(a == b, "Types of A and B must be the same");
		LIBRAPID_ASSERT(a == c, "Output type must be the same as input types");

		switch (::librapid::min(a, b, c)) {
			case CUDA_R_16F:
			case CUDA_C_16F: // 16-bit -> 16-bit
				return selector({CUBLAS_COMPUTE_16F, CUDA_R_16F}, {CUBLAS_COMPUTE_32F, CUDA_R_32F});
			case CUDA_R_32F:
			case CUDA_C_32F: // 32-bit -> [ fast: 16-bit, precise: 32-bit ]
				return selector({CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F},
								{CUBLAS_COMPUTE_32F, CUDA_R_32F});
			case CUDA_R_64F:
			case CUDA_C_64F: // 64-bit -> 64-bit
				return selector({CUBLAS_COMPUTE_64F, CUDA_R_64F}, {CUBLAS_COMPUTE_64F, CUDA_R_64F});
			case CUDA_R_32I:
			case CUDA_C_32I: // 32-bit -> 32-bit
				return selector({CUBLAS_COMPUTE_32I, CUDA_R_32I}, {CUBLAS_COMPUTE_32I, CUDA_R_32I});
			default: {
				LIBRAPID_ASSERT(false, "Invalid input types to CuBLAS gemm");
				return {CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F};
			}
		}
	}

	template<typename Int, typename Alpha, typename A, typename B, typename Beta, typename C>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, A *a, Int lda, B *b,
			  Int ldb, Beta beta, C *c, Int ldc, backend::CUDA) {
		if constexpr (typetraits::IsBlasType<A>::value && typetraits::IsBlasType<B>::value &&
					  typetraits::IsBlasType<C>::value) {
			cublasLtMatmulDesc_t operationDescriptor = nullptr;
			cublasLtMatrixLayout_t descriptorA = nullptr, descriptorB = nullptr,
								   descriptorC	  = nullptr;
			cublasLtMatmulPreference_t preference = NULL;

			constexpr int maxHeuristicResults									  = 32;
			int returnedResults													  = 0;
			cublasLtMatmulHeuristicResult_t heuristicResults[maxHeuristicResults] = {};

			cublasOperation_t cublasTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t cublasTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

			cudaDataType_t cudaTypeA = typetraits::TypeInfo<A>::CudaType;
			cudaDataType_t cudaTypeB = typetraits::TypeInfo<B>::CudaType;
			cudaDataType_t cudaTypeC = typetraits::TypeInfo<C>::CudaType;

			// Create operation descriptors
			auto [computeType, scaleType] = cublasGemmComputeType(cudaTypeA, cudaTypeB, cudaTypeC);
			cublasSafeCall(cublasLtMatmulDescCreate(&operationDescriptor, computeType, scaleType));
			cublasSafeCall(cublasLtMatmulDescSetAttribute(operationDescriptor,
														  CUBLASLT_MATMUL_DESC_TRANSA,
														  &cublasTransA,
														  sizeof(cublasTransA)));
			cublasSafeCall(cublasLtMatmulDescSetAttribute(operationDescriptor,
														  CUBLASLT_MATMUL_DESC_TRANSB,
														  &cublasTransB,
														  sizeof(cublasTransB)));

			// Create matrix descriptors
			cublasSafeCall(cublasLtMatrixLayoutCreate(
			  &descriptorA, cudaTypeA, !transA ? m : k, !transA ? k : m, lda));
			cublasSafeCall(cublasLtMatrixLayoutCreate(
			  &descriptorB, cudaTypeB, !transB ? k : n, !transB ? n : k, ldb));
			cublasSafeCall(cublasLtMatrixLayoutCreate(&descriptorC, cudaTypeC, m, n, ldc));

			// Set layout attributes
			const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
			cublasSafeCall(cublasLtMatrixLayoutSetAttribute(
			  descriptorA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
			cublasSafeCall(cublasLtMatrixLayoutSetAttribute(
			  descriptorB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
			cublasSafeCall(cublasLtMatrixLayoutSetAttribute(
			  descriptorC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

			// Create preference handle
			cublasSafeCall(cublasLtMatmulPreferenceCreate(&preference));
			cublasSafeCall(
			  cublasLtMatmulPreferenceSetAttribute(preference,
												   CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
												   &global::cublasLtWorkspaceSize,
												   sizeof(global::cublasLtWorkspaceSize)));

			// Find the best algorithm to use for the given problem
			cublasSafeCall(cublasLtMatmulAlgoGetHeuristic(global::cublasLtHandle,
														  operationDescriptor,
														  descriptorA,
														  descriptorB,
														  descriptorC,
														  descriptorC,
														  preference,
														  maxHeuristicResults,
														  &heuristicResults[0],
														  &returnedResults));

			LIBRAPID_ASSERT(returnedResults != 0, "Invalid matrices for GEMM. No algorithm found.");

			size_t i = 0;
			for (; i < returnedResults; ++i) {
				if (heuristicResults[i].state == CUBLAS_STATUS_SUCCESS) {
					cublasSafeCall(cublasLtMatmul(global::cublasLtHandle,
												  operationDescriptor,
												  &alpha,
												  a,
												  descriptorA,
												  b,
												  descriptorB,
												  &beta,
												  c,
												  descriptorC,
												  c,
												  descriptorC,
												  &heuristicResults[i].algo,
												  global::cublasLtWorkspace,
												  global::cublasLtWorkspaceSize,
												  global::cudaStream));
					break;
				}
			}

			LIBRAPID_ASSERT(i != returnedResults, "Invalid matrices for GEMM. No algorithm found.");

			cublasSafeCall(cublasLtMatmulPreferenceDestroy(preference));
			cublasSafeCall(cublasLtMatrixLayoutDestroy(descriptorA));
			cublasSafeCall(cublasLtMatrixLayoutDestroy(descriptorB));
			cublasSafeCall(cublasLtMatrixLayoutDestroy(descriptorC));
			cublasSafeCall(cublasLtMatmulDescDestroy(operationDescriptor));
		} else {
			jitify::Program program = global::jitCache.program(cuda::loadKernel(
			  fmt::format("{}/include/librapid/array/linalg/level3/gemm", LIBRAPID_SOURCE), false));

			size_t TS = 32;

			dim3 threadsPerBlock(TS, TS);
			dim3 numBlocks((n + TS - 1) / TS, (m + TS - 1) / TS);

			jitifyCall(program.kernel("gemm")
						 .instantiate(jitify::reflection::Type<Int>(),
									  jitify::reflection::Type<Alpha>(),
									  jitify::reflection::Type<A>(),
									  jitify::reflection::Type<Beta>(),
									  jitify::reflection::Type<B>(),
									  jitify::reflection::Type<C>())
						 .configure(numBlocks, threadsPerBlock, 0, global::cudaStream)
						 .launch(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
		}
	}

#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::linalg

#endif // LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP