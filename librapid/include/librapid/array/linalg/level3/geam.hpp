#ifndef LIBRAPID_ARRAY_LINALG_LEVEL3_GEAM_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL3_GEAM_HPP

namespace librapid::linalg {
	template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
			 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
			 typename StorageAllocatorC, typename Scalar>
	void geam(const array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>> &a,
			  Scalar alpha,
			  const array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>> &b,
			  Scalar beta,
			  array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		c = a * alpha + b * beta;
	}

	template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
			 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
			 typename StorageAllocatorC, typename Scalar>
	void geam(const array::Transpose<
				array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>>> &a,
			  Scalar alpha,
			  const array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>> &b,
			  Scalar beta,
			  array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		// Eval before returning to avoid slow evaluation
		// see https://librapid.readthedocs.io/en/latest/performance/performance.html

		const auto &dataA = a.array();

		c = array::Transpose(dataA, {1, 0}, alpha).eval() + b * beta;
	}

	template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
			 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
			 typename StorageAllocatorC, typename Scalar>
	void geam(const array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>> &a,
			  Scalar alpha,
			  const array::Transpose<
				array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>>> &b,
			  Scalar beta,
			  array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");

		// Eval before returning to avoid slow evaluation
		// see https://librapid.readthedocs.io/en/latest/performance/performance.html

		const auto &dataB = b.array();

		c = a * alpha + array::Transpose(dataB, {1, 0}, beta).eval();
	}

	template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
			 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
			 typename StorageAllocatorC, typename Scalar>
	void geam(const array::Transpose<
				array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>>> &a,
			  Scalar alpha,
			  const array::Transpose<
				array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>>> &b,
			  Scalar beta,
			  array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		// Eval before returning to avoid slow evaluation
		// see https://librapid.readthedocs.io/en/latest/performance/performance.html

		// alpha * a^T + beta * b^T = (alpha * a + beta * b)^T

		const auto &dataA = a.array();
		const auto &dataB = b.array();

		c = transpose((dataA * alpha + dataB * beta).eval());
	}

#if defined(LIBRAPID_HAS_CUDA)

	template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC,
			 typename Scalar>
	void geam(const array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>> &a, Scalar alpha,
			  const array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>> &b, Scalar beta,
			  array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		c = a * alpha + b * beta;
	}

	template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC,
			 typename Scalar>
	void
	geam(const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>>> &a,
		 Scalar alpha, const array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>> &b,
		 Scalar beta, array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		const auto &dataA = a.array();

		c = array::Transpose(dataA, {1, 0}, alpha).eval() + b * beta;
	}

	template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC,
			 typename Scalar>
	void
	geam(const array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>> &a, Scalar alpha,
		 const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>>> &b,
		 Scalar beta, array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		const auto &dataB = b.array();

		c = a * alpha + array::Transpose(dataB, {1, 0}, beta).eval();
	}

	template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC,
			 typename Scalar>
	void
	geam(const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>>> &a,
		 Scalar alpha,
		 const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>>> &b,
		 Scalar beta, array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
		LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");
		LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");
		LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");
		LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");
		LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");

		const auto &dataA = a.array();
		const auto &dataB = b.array();

		c = transpose((dataA * alpha + dataB * beta).eval());
	}

#	define LIBRAPID_CUDA_GEAM_IMPL(SCALAR, PREFIX)                                                \
		template<typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC, typename Scalar>   \
		void geam(const array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>> &a,                 \
				  Scalar alpha,                                                                    \
				  const array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>> &b,                 \
				  Scalar beta,                                                                     \
				  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                     \
			LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");                    \
			LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");       \
			LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");                   \
			LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");                \
			LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");                \
                                                                                                   \
			auto *__restrict dataA = a.storage().begin().get();                                    \
			auto *__restrict dataB = b.storage().begin().get();                                    \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_N,                                       \
												CUBLAS_OP_N,                                       \
												a.shape()[0],                                      \
												a.shape()[1],                                      \
												&alpha,                                            \
												dataA,                                             \
												a.shape()[0],                                      \
												&beta,                                             \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}                                                                                          \
                                                                                                   \
		template<typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC, typename Scalar>   \
		void geam(                                                                                 \
		  const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>>> &a,       \
		  Scalar alpha,                                                                            \
		  const array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>> &b,                         \
		  Scalar beta,                                                                             \
		  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                             \
			LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");                    \
			LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");       \
			LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");                   \
			LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");                \
			LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");                \
                                                                                                   \
			auto *__restrict dataA = a.array().storage().begin().get();                            \
			auto *__restrict dataB = b.storage().begin().get();                                    \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_T,                                       \
												CUBLAS_OP_N,                                       \
												a.shape()[1],                                      \
												a.shape()[0],                                      \
												&alpha,                                            \
												dataA,                                             \
												a.shape()[0],                                      \
												&beta,                                             \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}                                                                                          \
                                                                                                   \
		template<typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC, typename Scalar>   \
		void geam(                                                                                 \
		  const array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>> &a,                         \
		  Scalar alpha,                                                                            \
		  const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>>> &b,       \
		  Scalar beta,                                                                             \
		  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                             \
			LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");                    \
			LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");       \
			LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");                   \
			LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");                \
			LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");                \
                                                                                                   \
			auto *__restrict dataA = a.storage().begin().get();                                    \
			auto *__restrict dataB = b.array().storage().begin().get();                            \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_N,                                       \
												CUBLAS_OP_T,                                       \
												a.shape()[0],                                      \
												a.shape()[1],                                      \
												&alpha,                                            \
												dataA,                                             \
												a.shape()[0],                                      \
												&beta,                                             \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}                                                                                          \
                                                                                                   \
		template<typename ShapeTypeA, typename ShapeTypeB, typename ShapeTypeC, typename Scalar>   \
		void geam(                                                                                 \
		  const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>>> &a,       \
		  Scalar alpha,                                                                            \
		  const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>>> &b,       \
		  Scalar beta,                                                                             \
		  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                             \
			LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");                    \
			LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");       \
			LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");                   \
			LIBRAPID_ASSERT(&a != &c, "Input and output arrays must be different");                \
			LIBRAPID_ASSERT(&b != &c, "Input and output arrays must be different");                \
                                                                                                   \
			auto *__restrict dataA = a.array().storage().begin().get();                            \
			auto *__restrict dataB = b.array().storage().begin().get();                            \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_T,                                       \
												CUBLAS_OP_T,                                       \
												a.shape()[1],                                      \
												a.shape()[0],                                      \
												&alpha,                                            \
												dataA,                                             \
												a.shape()[0],                                      \
												&beta,                                             \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}

	LIBRAPID_CUDA_GEAM_IMPL(float, S)
	LIBRAPID_CUDA_GEAM_IMPL(double, D)
	LIBRAPID_CUDA_GEAM_IMPL(Complex<float>, C)
	LIBRAPID_CUDA_GEAM_IMPL(Complex<double>, Z)

#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::linalg

#endif // LIBRAPID_ARRAY_LINALG_LEVEL3_GEAM_HPP