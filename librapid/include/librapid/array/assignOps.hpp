#ifndef LIBRAPID_ARRAY_ASSIGN_OPS_HPP
#define LIBRAPID_ARRAY_ASSIGN_OPS_HPP

namespace librapid::detail {
	// All assignment operators are forward declared in "forward.hpp" so they can be used
	// elsewhere. They are defined here.

	/// Trivial array assignment operator -- assignment can be done with a single vectorised
	/// loop over contiguous data.
	/// \tparam ShapeType_ The shape type of the array container
	/// \tparam StorageType_ The storage type of the array container
	/// \tparam Functor_ The function type
	/// \tparam Args The argument types of the function
	/// \param lhs The array container to assign to
	/// \param function The function to assign
	template<typename ShapeType_, typename StorageType_, typename Functor_, typename... Args>
	LIBRAPID_ALWAYS_INLINE void
	assign(ArrayContainer<ShapeType_, StorageType_> &lhs,
		   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function) {
		using Scalar				  = typename ArrayContainer<ShapeType_, StorageType_>::Scalar;
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

	template<typename ShapeType_, typename StorageType_, typename Functor_, typename... Args>
	LIBRAPID_ALWAYS_INLINE void
	assignParallel(ArrayContainer<ShapeType_, StorageType_> &lhs,
				   const detail::Function<Descriptor::Trivial, Functor_, Args...> &function) {
		using Scalar				  = typename ArrayContainer<ShapeType_, StorageType_>::Scalar;
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
} // namespace librapid::detail

#endif // LIBRAPID_ARRAY_ASSIGN_OPS_HPP