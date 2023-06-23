#ifndef LIBRAPID_ARRAY_LINALG_LEVEL3_GEAM_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL3_GEAM_HPP

namespace librapid {
	namespace linalg {
#define GEAM_VALIDATION                                                                            \
	LIBRAPID_ASSERT(a.shape() == b.shape(), "Input shapes must match");                            \
	LIBRAPID_ASSERT(a.ndim() == 2, "Input array must be a Matrix (2D)");                           \
	LIBRAPID_ASSERT(a.shape() == c.shape(), "Output shape must match input shapes");               \
	LIBRAPID_ASSERT((void *)&a != (void *)&c, "Input and output arrays must be different");        \
	LIBRAPID_ASSERT((void *)&b != (void *)&c, "Input and output arrays must be different")

		template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
				 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
				 typename StorageAllocatorC, typename Alpha, typename Beta>
		void
		geam(const array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>> &a,
			 Alpha alpha,
			 const array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>> &b,
			 Beta beta,
			 array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
			GEAM_VALIDATION;

			c = a * static_cast<StorageScalar>(alpha) + b * static_cast<StorageScalar>(beta);
		}

		template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
				 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
				 typename StorageAllocatorC, typename Alpha, typename Beta>
		void
		geam(const array::Transpose<
			   array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>>> &a,
			 Alpha alpha,
			 const array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>> &b,
			 Beta beta,
			 array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
			GEAM_VALIDATION;

			// Eval before returning to avoid slow evaluation
			// see https://librapid.readthedocs.io/en/latest/performance/performance.html

			const auto &dataA = a.array();

			c = array::Transpose(dataA, {1, 0}, static_cast<StorageScalar>(alpha)).eval() +
				b * static_cast<StorageScalar>(beta);
		}

		template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
				 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
				 typename StorageAllocatorC, typename Alpha, typename Beta>
		void
		geam(const array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>> &a,
			 Alpha alpha,
			 const array::Transpose<
			   array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>>> &b,
			 Beta beta,
			 array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
			GEAM_VALIDATION;

			// Eval before returning to avoid slow evaluation
			// see https://librapid.readthedocs.io/en/latest/performance/performance.html

			const auto &dataB = b.array();

			c = a * static_cast<StorageScalar>(alpha) +
				array::Transpose(dataB, {1, 0}, static_cast<StorageScalar>(beta)).eval();
		}

		template<typename StorageScalar, typename ShapeTypeA, typename StorageAllocatorA,
				 typename ShapeTypeB, typename StorageAllocatorB, typename ShapeTypeC,
				 typename StorageAllocatorC, typename Alpha, typename Beta>
		void
		geam(const array::Transpose<
			   array::ArrayContainer<ShapeTypeA, Storage<StorageScalar, StorageAllocatorA>>> &a,
			 Alpha alpha,
			 const array::Transpose<
			   array::ArrayContainer<ShapeTypeB, Storage<StorageScalar, StorageAllocatorB>>> &b,
			 Beta beta,
			 array::ArrayContainer<ShapeTypeC, Storage<StorageScalar, StorageAllocatorC>> &c) {
			GEAM_VALIDATION;

			// Eval before returning to avoid slow evaluation
			// see https://librapid.readthedocs.io/en/latest/performance/performance.html

			// alpha * a^T + beta * b^T = (alpha * a + beta * b)^T

			const auto &dataA = a.array();
			const auto &dataB = b.array();

			c = transpose(
			  (dataA * static_cast<StorageScalar>(alpha) + dataB * static_cast<StorageScalar>(beta))
				.eval());
		}

#if defined(LIBRAPID_HAS_OPENCL)

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

		template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB,
				 typename ShapeTypeC, typename Alpha, typename Beta>
		void geam(const array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>> &a,
				  Alpha alpha,
				  const array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>> &b, Beta beta,
				  array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
			GEAM_VALIDATION;

			c = a * static_cast<StorageScalar>(alpha) + b * static_cast<StorageScalar>(beta);
		}

		template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB,
				 typename ShapeTypeC, typename Alpha, typename Beta>
		void geam(
		  const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>>> &a,
		  Alpha alpha, const array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>> &b,
		  Beta beta, array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
			GEAM_VALIDATION;

			const auto &dataA = a.array();

			c = array::Transpose(dataA, {1, 0}, static_cast<StorageScalar>(alpha)).eval() +
				b * static_cast<StorageScalar>(beta);
		}

		template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB,
				 typename ShapeTypeC, typename Alpha, typename Beta>
		void geam(
		  const array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>> &a, Alpha alpha,
		  const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>>> &b,
		  Beta beta, array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
			GEAM_VALIDATION;

			const auto &dataB = b.array();

			c = a * static_cast<StorageScalar>(alpha) +
				array::Transpose(dataB, {1, 0}, static_cast<StorageScalar>(beta)).eval();
		}

		template<typename StorageScalar, typename ShapeTypeA, typename ShapeTypeB,
				 typename ShapeTypeC, typename Alpha, typename Beta>
		void geam(
		  const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<StorageScalar>>> &a,
		  Alpha alpha,
		  const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<StorageScalar>>> &b,
		  Beta beta, array::ArrayContainer<ShapeTypeC, CudaStorage<StorageScalar>> &c) {
			GEAM_VALIDATION;

			const auto &dataA = a.array();
			const auto &dataB = b.array();

			c = transpose(
			  (dataA * static_cast<StorageScalar>(alpha) + dataB * static_cast<StorageScalar>(beta))
				.eval());
		}

#	define LIBRAPID_CUDA_GEAM_IMPL(SCALAR, PREFIX)                                                \
		template<typename ShapeTypeA,                                                              \
				 typename ShapeTypeB,                                                              \
				 typename ShapeTypeC,                                                              \
				 typename Alpha,                                                                   \
				 typename Beta>                                                                    \
		void geam(const array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>> &a,                 \
				  Alpha alpha,                                                                     \
				  const array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>> &b,                 \
				  Beta beta,                                                                       \
				  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                     \
			GEAM_VALIDATION;                                                                       \
                                                                                                   \
			auto *__restrict dataA = a.storage().begin().get();                                    \
			auto *__restrict dataB = b.storage().begin().get();                                    \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			auto alphaTmp = static_cast<SCALAR>(alpha);                                            \
			auto betaTmp  = static_cast<SCALAR>(beta);                                             \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_N,                                       \
												CUBLAS_OP_N,                                       \
												a.shape()[0],                                      \
												a.shape()[1],                                      \
												&alphaTmp,                                         \
												dataA,                                             \
												a.shape()[0],                                      \
												&betaTmp,                                          \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}                                                                                          \
                                                                                                   \
		template<typename ShapeTypeA,                                                              \
				 typename ShapeTypeB,                                                              \
				 typename ShapeTypeC,                                                              \
				 typename Alpha,                                                                   \
				 typename Beta>                                                                    \
		void geam(                                                                                 \
		  const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>>> &a,       \
		  Alpha alpha,                                                                             \
		  const array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>> &b,                         \
		  Beta beta,                                                                               \
		  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                             \
			GEAM_VALIDATION;                                                                       \
                                                                                                   \
			auto *__restrict dataA = a.array().storage().begin().get();                            \
			auto *__restrict dataB = b.storage().begin().get();                                    \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			auto alphaTmp = static_cast<SCALAR>(alpha);                                            \
			auto betaTmp  = static_cast<SCALAR>(beta);                                             \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_T,                                       \
												CUBLAS_OP_N,                                       \
												a.shape()[1],                                      \
												a.shape()[0],                                      \
												&alphaTmp,                                         \
												dataA,                                             \
												a.shape()[0],                                      \
												&betaTmp,                                          \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}                                                                                          \
                                                                                                   \
		template<typename ShapeTypeA,                                                              \
				 typename ShapeTypeB,                                                              \
				 typename ShapeTypeC,                                                              \
				 typename Alpha,                                                                   \
				 typename Beta>                                                                    \
		void geam(                                                                                 \
		  const array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>> &a,                         \
		  Alpha alpha,                                                                             \
		  const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>>> &b,       \
		  Beta beta,                                                                               \
		  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                             \
			GEAM_VALIDATION;                                                                       \
                                                                                                   \
			auto *__restrict dataA = a.storage().begin().get();                                    \
			auto *__restrict dataB = b.array().storage().begin().get();                            \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			auto alphaTmp = static_cast<SCALAR>(alpha);                                            \
			auto betaTmp  = static_cast<SCALAR>(beta);                                             \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_N,                                       \
												CUBLAS_OP_T,                                       \
												a.shape()[0],                                      \
												a.shape()[1],                                      \
												&alphaTmp,                                         \
												dataA,                                             \
												a.shape()[0],                                      \
												&betaTmp,                                          \
												dataB,                                             \
												b.shape()[0],                                      \
												dataC,                                             \
												c.shape()[0]));                                    \
		}                                                                                          \
                                                                                                   \
		template<typename ShapeTypeA,                                                              \
				 typename ShapeTypeB,                                                              \
				 typename ShapeTypeC,                                                              \
				 typename Alpha,                                                                   \
				 typename Beta>                                                                    \
		void geam(                                                                                 \
		  const array::Transpose<array::ArrayContainer<ShapeTypeA, CudaStorage<SCALAR>>> &a,       \
		  Alpha alpha,                                                                             \
		  const array::Transpose<array::ArrayContainer<ShapeTypeB, CudaStorage<SCALAR>>> &b,       \
		  Beta beta,                                                                               \
		  array::ArrayContainer<ShapeTypeC, CudaStorage<SCALAR>> &c) {                             \
			GEAM_VALIDATION;                                                                       \
                                                                                                   \
			auto *__restrict dataA = a.array().storage().begin().get();                            \
			auto *__restrict dataB = b.array().storage().begin().get();                            \
			auto *__restrict dataC = c.storage().begin().get();                                    \
                                                                                                   \
			auto alphaTmp = static_cast<SCALAR>(alpha);                                            \
			auto betaTmp  = static_cast<SCALAR>(beta);                                             \
                                                                                                   \
			cublasSafeCall(cublas##PREFIX##geam(global::cublasHandle,                              \
												CUBLAS_OP_T,                                       \
												CUBLAS_OP_T,                                       \
												a.shape()[1],                                      \
												a.shape()[0],                                      \
												&alphaTmp,                                         \
												dataA,                                             \
												a.shape()[0],                                      \
												&betaTmp,                                          \
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
	}  // namespace linalg

	namespace typetraits {
		template<typename Descriptor1, typename Descriptor2, typename Descriptor3,
				 typename TransposeType1, typename TransposeType2, typename ScalarType1,
				 typename ScalarType2>
		struct HasCustomEval<
		  detail::Function<Descriptor1, detail::Plus,
						   detail::Function<Descriptor2, detail::Multiply,
											array::Transpose<TransposeType1>, ScalarType1>,
						   detail::Function<Descriptor3, detail::Multiply,
											array::Transpose<TransposeType2>, ScalarType2>>>
				: std::true_type {};
	}; // namespace typetraits

	namespace detail {
		// aT * b + cT * d
		template<typename ShapeType, typename DestinationStorageType, typename Descriptor1,
				 typename Descriptor2, typename Descriptor3, typename TransposeType1,
				 typename TransposeType2, typename ScalarType1, typename ScalarType2>
		LIBRAPID_ALWAYS_INLINE void assign(
		  array::ArrayContainer<ShapeType, DestinationStorageType> &destination,
		  const Function<
			Descriptor1, detail::Plus,
			Function<Descriptor2, detail::Multiply, array::Transpose<TransposeType1>, ScalarType1>,
			Function<Descriptor3, detail::Multiply, array::Transpose<TransposeType2>, ScalarType2>>
			&function) {
			// Since GEAM only applies to matrices, we must check that we can actually use it given
			// the input matrices. If we can't, we fall back to the default implementation.

			using Scalar = typename DestinationStorageType::Scalar;

			bool canUseGeam	 = true;
			auto left		 = std::get<0>(function.args());
			auto leftMat	 = std::get<0>(left.args());
			auto leftScalar	 = std::get<1>(left.args());
			auto right		 = std::get<1>(function.args());
			auto rightMat	 = std::get<0>(right.args());
			auto rightScalar = std::get<1>(right.args());

			if (leftMat.ndim() != 2 || rightMat.ndim() != 2 || destination.ndim() != 2) {
				canUseGeam = false;
			}

			if (leftMat.shape() != rightMat.shape() || leftMat.shape() != destination.shape()) {
				canUseGeam = false;
			}

			if (canUseGeam) {
				linalg::geam(leftMat,
							 static_cast<Scalar>(leftScalar),
							 rightMat,
							 static_cast<Scalar>(rightScalar),
							 destination);
			} else {
				auto axes1	= leftMat.axes();
				auto alpha	= leftMat.alpha() * static_cast<Scalar>(leftScalar);
				auto axes2	= rightMat.axes();
				auto beta	= rightMat.alpha() * static_cast<Scalar>(rightScalar);
				destination = array::Transpose(leftMat.array(), axes1, alpha).eval() +
							  array::Transpose(rightMat.array(), axes2, beta).eval();
			}
		}
	} // namespace detail
} // namespace librapid

#endif // LIBRAPID_ARRAY_LINALG_LEVEL3_GEAM_HPP