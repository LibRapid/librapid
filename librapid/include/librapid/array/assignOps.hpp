#ifndef LIBRAPID_ARRAY_ASSIGN_OPS_HPP
#define LIBRAPID_ARRAY_ASSIGN_OPS_HPP

namespace librapid::detail {
	// All assignment operators are forward declared in "forward.hpp" so they can be used
	// elsewhere. They are defined here.

	/// Trivial array assignment operator -- assignment can be done with a single vectorised
	/// loop over contiguous data.
	/// \tparam ShapeType_ The shape type of the array container
	/// \tparam StorageScalar The scalar type of the storage object
	/// \tparam StorageAllocator The Allocator of the Storage object
	/// \tparam Functor_ The function type
	/// \tparam Args The argument types of the function
	/// \param lhs The array container to assign to
	/// \param function The function to assign
	template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
			 typename Functor_, typename... Args>
	LIBRAPID_ALWAYS_INLINE void
	assign(ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
		   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function) {
		using Scalar =
		  typename ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>>::Scalar;
		constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;

		const int64_t size		 = function.shape().size();
		const int64_t vectorSize = size - (size % packetWidth);

		// Ensure the function can actually be assigned to the array container
		static_assert(typetraits::IsSame<Scalar, typename std::decay_t<decltype(function)>::Scalar>,
					  "Function return type must be the same as the array container's scalar type");
		LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

		for (int64_t index = 0; index < vectorSize; index += packetWidth) {
			lhs.writePacket(index, function.packet(index));
		}

		// Assign the remaining elements
		for (int64_t index = vectorSize; index < size; ++index) {
			lhs.write(index, function.scalar(index));
		}
	}

	/// Trivial assignment with parallel execution
	/// \tparam ShapeType_
	/// \tparam StorageScalar
	/// \tparam StorageAllocator
	/// \tparam Functor_
	/// \tparam Args
	/// \param lhs
	/// \param function
	/// \see assign(ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
	///				const detail::Function<Descriptor::Trivial, Functor_, Args...> &function)
	template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
			 typename Functor_, typename... Args>
	LIBRAPID_ALWAYS_INLINE void
	assignParallel(ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
				   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function) {
		using Scalar =
		  typename ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>>::Scalar;
		constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;

		const int64_t size		 = function.shape().size();
		const int64_t vectorSize = size - (size % packetWidth);

		// Ensure the function can actually be assigned to the array container
		static_assert(typetraits::IsSame<Scalar, typename std::decay_t<decltype(function)>::Scalar>,
					  "Function return type must be the same as the array container's scalar type");
		LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

#pragma omp parallel for shared(vectorSize, lhs, function) default(none)                           \
  num_threads(global::numThreads)
		for (int64_t index = 0; index < vectorSize; index += packetWidth) {
			lhs.writePacket(index, function.packet(index));
		}

		// Assign the remaining elements
		for (int64_t index = vectorSize; index < size; ++index) {
			lhs.write(index, function.scalar(index));
		}
	}

#if defined(LIBRAPID_HAS_CUDA)

	template<typename ShapeType_, typename StorageScalar, typename Functor_, typename... Args>
	LIBRAPID_ALWAYS_INLINE void
	assign(ArrayContainer<ShapeType_, CudaStorage<StorageScalar>> &lhs,
		   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function) {
		// Unfortunately, as we are not generating the kernels at runtime, we can't use
		// temporary-free evaluation. Instead, we must recursively evaluate each sub-operation
		// until a final result is computed

		constexpr const char *filename	 = typetraits::TypeInfo<Functor_>::filename;
		constexpr const char *kernelName = typetraits::TypeInfo<Functor_>::kernelName;
		using Scalar = typename ArrayContainer<ShapeType_, CudaStorage<StorageScalar>>::Scalar;

		runKernel<Scalar, Scalar, Scalar>(filename,
										  kernelName,
										  function.shape().size(),
										  function.shape().size(),
										  lhs.storage().begin(),
										  std::get<0>(function.args()).storage().begin(),
										  std::get<1>(function.args()).storage().begin());
	}

#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::detail

#endif // LIBRAPID_ARRAY_ASSIGN_OPS_HPP