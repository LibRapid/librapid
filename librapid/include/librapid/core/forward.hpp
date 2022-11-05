#ifndef LIBRAPID_CORE_FORWARD_HPP
#define LIBRAPID_CORE_FORWARD_HPP

namespace librapid {
	template<typename Scalar_, typename Allocator_>
	class Storage;

	template<typename Scalar_>
	class CudaStorage;

	template<typename ShapeType_, typename StorageType_>
	class ArrayContainer;

	namespace detail {
		/// \brief Identifies which type of function is being used
		enum class Descriptor {
			Trivial,   /// Operation is trivial and can be done with a vectorised loop
			Transpose, /// Operation is a matrix/array transposition
			Matmul	   /// Operation is a matrix/array multiplication
		};

		template<Descriptor desc, typename Functor_, typename... Args>
		class Function;

		template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
			   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function);

		template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assignParallel(ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
					   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function);

#if defined(LIBRAPID_HAS_CUDA)
		template<typename ShapeType_, typename StorageScalar,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(ArrayContainer<ShapeType_, CudaStorage<StorageScalar>> &lhs,
			   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function);

#endif // LIBRAPID_HAS_CUDA
	} // namespace detail
} // namespace librapid

#endif // LIBRAPID_CORE_FORWARD_HPP