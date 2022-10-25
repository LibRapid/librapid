#ifndef LIBRAPID_CORE_FORWARD_HPP
#define LIBRAPID_CORE_FORWARD_HPP

namespace librapid {
	template<typename Scalar_, typename Allocator_>
	class Storage;

	template<typename ShapeType_, typename StorageType_>
	class ArrayContainer;

	namespace detail {
		template<typename Functor_, typename... Args>
		class Function;

		/// Assign a function to an array container
		/// \tparam ShapeType_ The shape type of the array container
		/// \tparam StorageType_ The storage type of the array container
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		/// \param lhs The array container to assign to
		/// \param function The function to assign
		template<typename ShapeType_, typename StorageType_, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void assign(ArrayContainer<ShapeType_, StorageType_> &lhs,
										   const detail::Function<Functor_, Args...> &function);
	} // namespace detail
} // namespace librapid

#endif // LIBRAPID_CORE_FORWARD_HPP