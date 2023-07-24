#ifndef LIBRAPID_MATH_VECTOR_HPP
#define LIBRAPID_MATH_VECTOR_HPP

namespace librapid {
	namespace detail {
		template<typename T, size_t N, typename = void>
		struct VectorStorage;

		template<typename T, size_t N>
		struct VectorStorage<T, N, std::enable_if_t<(typetraits::TypeInfo<T>::packetWidth == 1)>> {
			static constexpr uint64_t length = N;
			T data[length];

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const auto &operator[](int64_t index) const {
				return data[index];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto &operator[](int64_t index) {
				return data[index];
			}
		};

		template<typename T, size_t N>
		struct VectorStorage<T, N, std::enable_if_t<(typetraits::TypeInfo<T>::packetWidth > 1)>> {
			using Packet						  = typename typetraits::TypeInfo<T>::Packet;
			static constexpr uint64_t packetWidth = typetraits::TypeInfo<T>::packetWidth;
			static constexpr uint64_t length =
			  (N + typetraits::TypeInfo<T>::packetWidth - 1) / typetraits::TypeInfo<T>::packetWidth;
			Packet data[length];

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index) const {
				const int64_t packetIndex  = index / packetWidth;
				const int64_t elementIndex = index % packetWidth;
				return data[packetIndex][elementIndex];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index) {
				const int64_t packetIndex  = index / packetWidth;
				const int64_t elementIndex = index % packetWidth;
				return data[packetIndex][elementIndex];
			}
		};

		template<typename T, size_t N, size_t... Indices>
		void vectorStorageAssigner(VectorStorage<T, N> &dst, const VectorStorage<T, N> &src,
								   std::index_sequence<Indices...>) {
			((dst[Indices] = src[Indices]), ...);
		}
	} // namespace detail

	template<typename ScalarType, size_t NumDims>
	class Vector {
	public:
		using Scalar					 = ScalarType;
		static constexpr uint64_t Dims	 = NumDims;
		using StorageType				 = detail::VectorStorage<Scalar, NumDims>;
		static constexpr uint64_t length = StorageType::length;

		Vector() = default;
		Vector(const Vector &other) = default;
		Vector(Vector &&other) noexcept = default;

		auto operator=(const Vector &other) -> Vector & = default;
		auto operator=(Vector &&other) noexcept -> Vector & = default;

		// template<typename T, size_t N>
		// Vector(const Vector<T, N> &other) {
		// 	detail::vectorStorageAssigner(m_data, other.m_data, std::make_index_sequence<length>());
		// }

	private:
		StorageType m_data;
	};
} // namespace librapid
#endif // LIBRAPID_MATH_VECTOR_HPP
