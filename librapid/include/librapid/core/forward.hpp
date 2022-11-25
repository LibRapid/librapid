#ifndef LIBRAPID_CORE_FORWARD_HPP
#define LIBRAPID_CORE_FORWARD_HPP

namespace librapid {
	template<typename Scalar_, typename Allocator_>
	class Storage;

	template<typename Scalar_, size_t... Dimensions>
	class FixedStorage;

	template<typename Scalar_>
	class CudaStorage;

	namespace array {
		template<typename ShapeType_, typename StorageType_>
		class ArrayContainer;
	}

	namespace detail {
		/// \brief Identifies which type of function is being used
		namespace descriptor {
			struct Trivial {};	 /// Operation is trivial and can be done with a vectorised loop
			struct Transpose {}; /// Operation is a matrix/array transposition
			struct Matmul {};	 /// Operation is a matrix/array multiplication
			struct Combined {};	 /// Operation is a combination of the above
		}						 // namespace descriptor

		template<typename desc, typename Functor_, typename... Args>
		class Function;

		template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function);

		template<typename ShapeType_, typename StorageScalar, size_t... StorageSize,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, FixedStorage<StorageScalar, StorageSize...>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function);

		template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void assignParallel(
		  array::ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
		  const detail::Function<descriptor::Trivial, Functor_, Args...> &function);

		template<typename ShapeType_, typename StorageScalar, size_t... StorageSize,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void assignParallel(
		  array::ArrayContainer<ShapeType_, FixedStorage<StorageScalar, StorageSize...>> &lhs,
		  const detail::Function<descriptor::Trivial, Functor_, Args...> &function);

#if defined(LIBRAPID_HAS_CUDA)
		template<typename ShapeType_, typename StorageScalar, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, CudaStorage<StorageScalar>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function);

#endif // LIBRAPID_HAS_CUDA
	}  // namespace detail
} // namespace librapid

#endif // LIBRAPID_CORE_FORWARD_HPP