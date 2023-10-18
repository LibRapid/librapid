#ifndef LIBRAPID_MATH_VECTOR_FORWARD_HPP
#define LIBRAPID_MATH_VECTOR_FORWARD_HPP

namespace librapid {
	namespace vectorDetail {
		template<typename T, uint64_t N>
		struct GenericVectorStorage;

		template<typename T, uint64_t N>
		struct SimdVectorStorage;

		template<typename T, uint64_t N>
		struct VectorStorageType {
			using type = std::conditional_t<(typetraits::TypeInfo<T>::packetWidth > 1),
											SimdVectorStorage<T, N>, GenericVectorStorage<T, N>>;
		};

		template<typename Storage0, typename Storage1>
		auto vectorStorageTypeMerger() {
			using Scalar0 = typename typetraits::TypeInfo<Storage0>::Scalar;
			using Scalar1 = typename typetraits::TypeInfo<Storage1>::Scalar;
			static constexpr uint64_t packetWidth0 = typetraits::TypeInfo<Scalar0>::packetWidth;
			static constexpr uint64_t packetWidth1 = typetraits::TypeInfo<Scalar1>::packetWidth;
			if constexpr (typetraits::TypeInfo<Storage0>::type == detail::LibRapidType::Scalar) {
				return Storage1 {};
			} else if constexpr (typetraits::TypeInfo<Storage1>::type ==
								 detail::LibRapidType::Scalar) {
				return Storage0 {};
			} else if constexpr (packetWidth0 > 1 && packetWidth1 > 1) {
				return SimdVectorStorage<typename Storage0::Scalar, Storage0::dims> {};
			} else {
				return GenericVectorStorage<typename Storage0::Scalar, Storage0::dims> {};
			}
		}

		template<typename T, uint64_t N>
		using VectorStorage = typename VectorStorageType<T, N>::type;

		template<typename Storage0, typename Storage1>
		using VectorStorageMerger = decltype(vectorStorageTypeMerger<Storage0, Storage1>());

		template<typename Derived>
		class VectorBase {
		public:
			using Scalar		 = typename typetraits::TypeInfo<Derived>::Scalar;
			using IndexType		 = typename typetraits::TypeInfo<Derived>::IndexType;
			using IndexTypeConst = typename typetraits::TypeInfo<Derived>::IndexTypeConst;
			using GetType		 = typename typetraits::TypeInfo<Derived>::GetType;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const auto &derived() const {
				return static_cast<const Derived &>(*this);
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto &derived() {
				return static_cast<Derived &>(*this);
			}

//			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE virtual Derived eval() const {
//				return derived();
//			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE virtual IndexTypeConst
			operator[](int64_t index) const {
				return derived()[index];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE virtual IndexType operator[](int64_t index) {
				return derived()[index];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst x() const {
				return derived()[0];
			}
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst y() const {
				return derived()[1];
			}
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst z() const {
				return derived()[2];
			}
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst w() const {
				return derived()[3];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType x() { return derived()[0]; }
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType y() { return derived()[1]; }
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType z() { return derived()[2]; }
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType w() { return derived()[3]; }

			template<typename T_, typename Char, typename Ctx>
			void str(const fmt::formatter<T_, Char> &formatter, Ctx &ctx) const {
				derived().str(formatter, ctx);
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE virtual GetType _get(uint64_t index) const {
				return derived()._get(index);
			}
		};
	} // namespace vectorDetail

	template<typename ScalarType, uint64_t NumDims>
	class Vector;

	namespace vectorDetail {
		template<typename LHS, typename RHS, typename Op>
		struct BinaryVecOp;

		template<typename Val, typename Op>
		struct UnaryVecOp;

		template<typename Scalar, uint64_t N, typename LHS, typename RHS, typename Op,
				 size_t... Indices>
		LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
											   const BinaryVecOp<LHS, RHS, Op> &src,
											   std::index_sequence<Indices...>);

		template<typename Scalar, uint64_t N, typename Val, typename Op, size_t... Indices>
		LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
											   const UnaryVecOp<Val, Op> &src,
											   std::index_sequence<Indices...>);

		template<typename Scalar, uint64_t N, typename LHS, typename RHS, typename Op>
		LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst,
										   const BinaryVecOp<LHS, RHS, Op> &src);

		template<typename Scalar, uint64_t N, typename Val, typename Op>
		LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst, const UnaryVecOp<Val, Op> &src);
	} // namespace vectorDetail

	namespace typetraits {
		LIBRAPID_DEFINE_AS_TYPE(typename ScalarType COMMA uint64_t NumDims,
								Vector<ScalarType COMMA NumDims>);

		LIBRAPID_DEFINE_AS_TYPE(typename LHS COMMA typename RHS COMMA typename Op,
								vectorDetail::BinaryVecOp<LHS COMMA RHS COMMA Op>);

		LIBRAPID_DEFINE_AS_TYPE(typename Val COMMA typename Op,
								vectorDetail::UnaryVecOp<Val COMMA Op>);
	} // namespace typetraits
} // namespace librapid

#endif // LIBRAPID_MATH_VECTOR_FORWARD_HPP