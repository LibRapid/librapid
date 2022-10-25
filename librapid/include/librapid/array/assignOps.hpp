#ifndef LIBRAPID_ARRAY_ASSIGN_OPS_HPP
#define LIBRAPID_ARRAY_ASSIGN_OPS_HPP

namespace librapid::detail {
	// All assignment operators are forward declared in "forward.hpp" so they can be used
	// elsewhere. They are defined here.

	template<typename ShapeType_, typename StorageType_, typename Functor_, typename... Args>
	LIBRAPID_ALWAYS_INLINE void assign(ArrayContainer<ShapeType_, StorageType_> &lhs,
									   const detail::Function<Functor_, Args...> &function) {
		using Scalar				  = typename ArrayContainer<ShapeType_, StorageType_>::Scalar;
		constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;

		int64_t size	   = function.shape().size();
		int64_t vectorSize = size - (size % packetWidth);

		bool multiThread = size > global::multithreadThreshold;

		// Ensure the function can actually be assigned to the array container
		static_assert(typetraits::IsSame<Scalar, typename Function<Functor_, Args...>::Scalar>,
					  "Function return type must be the same as the array container's scalar type");
		LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

		if (multiThread) {
#pragma omp parallel for shared(lhs, function, vectorSize) default(none) num_threads(global::numThreads)
			for (int64_t index = 0; index < vectorSize; index += packetWidth) {
				lhs.writePacket(index, function.packet(index));
			}

			// Assign the remaining elements
			for (int64_t index = vectorSize; index < size; ++index) {
				lhs.write(index, function.scalar(index));
			}
		} else {
			for (int64_t index = 0; index < vectorSize; index += packetWidth) {
				lhs.writePacket(index, function.packet(index));
			}

			// Assign the remaining elements
			for (int64_t index = vectorSize; index < size; ++index) {
				lhs.write(index, function.scalar(index));
			}
		}
	}
} // namespace librapid::detail

#endif // LIBRAPID_ARRAY_ASSIGN_OPS_HPP