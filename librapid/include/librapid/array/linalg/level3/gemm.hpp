#ifndef LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP

namespace librapid::linalg {
	template<typename Int, typename Alpha, typename A, typename Beta, typename B, typename C>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, A *a, Int lda, Beta beta,
			  B *b, Int ldb, C *c, Int ldc, backend::CPU backend = backend::CPU()) {
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
			  Beta beta, cl::Buffer b, Int ldb, cl::Buffer c, Int ldc, backend::OpenCL) {
		auto status = clblast::Gemm(clblast::Layout::kRowMajor,
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

		LIBRAPID_ASSERT(status == clblast::StatusCode::kSuccess, "clblast::Gemm failed");
	}

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

	/*
	template<typename Int, typename Alpha, typename Beta>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, std::shared_ptr<float> a,
			  Int lda, Beta beta, std::shared_ptr<float> b, Int ldb, std::shared_ptr<float> c,
			  Int ldc, backend::CUDA) {
		cublasSafeCall(cublasSgemm(global::cublasHandle,
								   (transA ? CUBLAS_OP_N : CUBLAS_OP_T),
								   (transB ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   k,
								   &alpha,
								   b.get(),
								   ldb,
								   a.get(),
								   lda,
								   &beta,
								   c.get(),
								   ldc));
	}

	template<typename Int, typename Alpha, typename Beta>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, std::shared_ptr<double> a,
			  Int lda, Beta beta, std::shared_ptr<double> b, Int ldb, std::shared_ptr<double> c,
			  Int ldc, backend::CUDA) {
		cublasSafeCall(cublasDgemm(global::cublasHandle,
								   (transA ? CUBLAS_OP_N : CUBLAS_OP_T),
								   (transB ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   k,
								   &alpha,
								   b.get(),
								   ldb,
								   a.get(),
								   lda,
								   &beta,
								   c.get(),
								   ldc));
	}
	 */

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
			case CUDA_C_32I:  // 32-bit -> 32-bit
				return selector({CUBLAS_COMPUTE_32I, CUDA_R_32I}, {CUBLAS_COMPUTE_32I, CUDA_R_32I});
			case CUDA_R_16BF: // <-- Invalid input types
			case CUDA_C_16BF:
			case CUDA_R_4I:
			case CUDA_C_4I:
			case CUDA_R_4U:
			case CUDA_C_4U:
			case CUDA_R_8I:
			case CUDA_C_8I:
			case CUDA_R_8U:
			case CUDA_C_8U:
			case CUDA_R_16I:
			case CUDA_C_16I:
			case CUDA_R_16U:
			case CUDA_C_16U:
			case CUDA_R_32U:
			case CUDA_C_32U:
			case CUDA_R_64I:
			case CUDA_C_64I:
			case CUDA_R_64U:
			case CUDA_C_64U:
			case CUDA_R_8F_E4M3:
			case CUDA_R_8F_E5M2: // Fallthrough
			{
				LIBRAPID_ASSERT(false, "Invalid input types to CuBLAS gemm");
				return {CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F};
			}
		}
	}

	//	template<typename Int, typename Alpha, typename A, typename Beta, typename B, typename C>
	//	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, std::shared_ptr<A> a,
	//			  Int lda, Beta beta, std::shared_ptr<B> b, Int ldb, std::shared_ptr<C> c, Int ldc,
	//			  backend::CUDA) {
	//		cublasLtMatmulDesc_t operationDescriptor = nullptr;
	//		cublasLtMatrixLayout_t descriptorA = nullptr, descriptorB = nullptr, descriptorC =
	// nullptr; 		cublasLtMatmulPreference_t preference = NULL;
	//
	//		constexpr size_t maxHeuristicResults								 = 32;
	//		int returnedResults													 = 0;
	//		cublasLtMatmulHeuristicResult_t heuristicResult[maxHeuristicResults] = {};
	//
	//		size_t workspaceSize = 1024 * 1024 * 1024;
	//
	//		cublasOperation_t cublasTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
	//		cublasOperation_t cublasTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
	//		cublasOperation_t cublasTransC = CUBLAS_OP_N;
	//
	//		cudaDataType_t cudaTypeA = CUDA_R_32F; // typetraits::TypeInfo<A>::CudaType;
	//		cudaDataType_t cudaTypeB = CUDA_R_32F; // typetraits::TypeInfo<B>::CudaType;
	//		cudaDataType_t cudaTypeC = CUDA_R_32F; // typetraits::TypeInfo<C>::CudaType;
	//
	//		// Create operation descriptors
	//		cublasSafeCall(cublasLtMatmulDescCreate(
	//		  &operationDescriptor, cublasGemmComputeType(cudaTypeA, cudaTypeB, cudaTypeC),
	// cudaTypeC)); 		cublasSafeCall(cublasLtMatmulDescSetAttribute( operationDescriptor,
	// CUBLASLT_MATMUL_DESC_TRANSA, &cublasTransA, sizeof(cublasTransA)));
	//		cublasSafeCall(cublasLtMatmulDescSetAttribute(
	//		  operationDescriptor, CUBLASLT_MATMUL_DESC_TRANSB, &cublasTransB,
	// sizeof(cublasTransB))); 		cublasSafeCall(cublasLtMatmulDescSetAttribute(
	// operationDescriptor, CUBLASLT_MATMUL_DESC_TRANSC, &cublasTransC, sizeof(cublasTransC)));
	//
	//		// Create matrix layouts
	//		cublasSafeCall(cublasLtMatrixLayoutCreate(&descriptorA, cudaTypeA, m, k, lda));
	//		cublasSafeCall(cublasLtMatrixLayoutCreate(&descriptorB, cudaTypeB, k, n, ldb));
	//		cublasSafeCall(cublasLtMatrixLayoutCreate(&descriptorC, cudaTypeC, m, n, ldc));
	//
	//		// Set layout attributes
	//		const cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
	//
	//		cublasSafeCall(cublasLtMatrixLayoutSetAttribute(
	//		  descriptorA, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
	//		cublasSafeCall(cublasLtMatrixLayoutSetAttribute(
	//		  descriptorB, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
	//		cublasSafeCall(cublasLtMatrixLayoutSetAttribute(
	//		  descriptorC, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
	//
	//		// Create preference handle
	//		cublasSafeCall(cublasLtMatmulPreferenceCreate(&preference));
	//		cublasSafeCall(
	//		  cublasLtMatmulPreferenceSetAttribute(preference,
	//											   CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
	//											   &workspaceSize,
	//											   sizeof(workspaceSize)));
	//
	//		// Set the size of the heuristic cache
	//		// cublasSafeCall(cublasLtHeuristicsCacheSetCapacity(1 << 20));
	//
	//		// Find the best algorithm to use for the given problem
	//		cublasSafeCall(cublasLtMatmulAlgoGetHeuristic(global::cublasLtHandle,
	//													  operationDescriptor,
	//													  descriptorA,
	//													  descriptorB,
	//													  descriptorC,
	//													  descriptorC,
	//													  preference,
	//													  maxHeuristicResults,
	//													  &heuristicResult[0],
	//													  &returnedResults));
	//
	//		LIBRAPID_ASSERT(returnedResults != 0, "Invalid matrices for GEMM. No algorithm found.");
	//
	//		cublasSafeCall(cublasLtMatmul(global::cublasLtHandle,
	//									  operationDescriptor,
	//									  &alpha,
	//									  a.get(),
	//									  descriptorA,
	//									  b.get(),
	//									  descriptorB,
	//									  &beta,
	//									  c.get(),
	//									  descriptorC,
	//									  c.get(),
	//									  descriptorC,
	//									  &heuristicResult[0].algo,
	//									  nullptr,
	//									  0,
	//									  global::cudaStream));
	//
	//		cublasSafeCall(cublasLtMatmulPreferenceDestroy(preference));
	//		cublasSafeCall(cublasLtMatrixLayoutDestroy(descriptorA));
	//		cublasSafeCall(cublasLtMatrixLayoutDestroy(descriptorB));
	//		cublasSafeCall(cublasLtMatrixLayoutDestroy(descriptorC));
	//		cublasSafeCall(cublasLtMatmulDescDestroy(operationDescriptor));
	//	}

	template<typename Int, typename Alpha, typename A, typename Beta, typename B, typename C>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, std::shared_ptr<A> a,
			  Int lda, Beta beta, std::shared_ptr<B> b, Int ldb, std::shared_ptr<C> c, Int ldc,
			  backend::CUDA) {
		cublasLtMatmulDesc_t operationDescriptor = nullptr;
		cublasLtMatrixLayout_t descriptorA = nullptr, descriptorB = nullptr, descriptorC = nullptr;
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
		cublasSafeCall(cublasLtMatmulDescSetAttribute(
		  operationDescriptor, CUBLASLT_MATMUL_DESC_TRANSA, &cublasTransA, sizeof(cublasTransA)));
		cublasSafeCall(cublasLtMatmulDescSetAttribute(
		  operationDescriptor, CUBLASLT_MATMUL_DESC_TRANSB, &cublasTransB, sizeof(cublasTransB)));

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
											  a.get(),
											  descriptorA,
											  b.get(),
											  descriptorB,
											  &beta,
											  c.get(),
											  descriptorC,
											  c.get(),
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
	}

#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::linalg

#endif // LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP