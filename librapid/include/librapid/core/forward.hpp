#ifndef LIBRAPID_CORE_FORWARD_HPP
#define LIBRAPID_CORE_FORWARD_HPP

namespace librapid {
	template<typename Scalar_, typename Allocator_>
	class Storage;

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

		template<Descriptor desc, typename ShapeType_, typename StorageType_, typename Functor_,
				 typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(ArrayContainer<ShapeType_, StorageType_> &lhs,
			   const detail::Function<desc, ShapeType_, Args...> &function);

		template<Descriptor desc, typename ShapeType_, typename StorageType_, typename Functor_,
				 typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assignParallel(ArrayContainer<ShapeType_, StorageType_> &lhs,
					   const detail::Function<desc, ShapeType_, Args...> &function);
	} // namespace detail
} // namespace librapid

#endif // LIBRAPID_CORE_FORWARD_HPP