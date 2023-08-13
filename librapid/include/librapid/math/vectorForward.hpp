#ifndef LIBRAPID_MATH_VECTOR_FORWARD_HPP
#define LIBRAPID_MATH_VECTOR_FORWARD_HPP

namespace librapid {
    namespace vectorDetail {
        template<typename T, size_t N>
        struct GenericVectorStorage;

        template<typename T, size_t N>
        struct SimdVectorStorage;

        template<typename T, size_t N>
        struct VectorStorageType {
            using type = std::conditional_t<(typetraits::TypeInfo<T>::packetWidth > 1),
                                            SimdVectorStorage<T, N>, GenericVectorStorage<T, N>>;
        };

        template<typename Storage0, typename Storage1>
        auto vectorStorageTypeMerger() {
            using Scalar0                        = typename typetraits::TypeInfo<Storage0>::Scalar;
            using Scalar1                        = typename typetraits::TypeInfo<Storage1>::Scalar;
            static constexpr size_t packetWidth0 = typetraits::TypeInfo<Scalar0>::packetWidth;
            static constexpr size_t packetWidth1 = typetraits::TypeInfo<Scalar1>::packetWidth;
            if constexpr (packetWidth0 > 1 && packetWidth1 > 1) {
                return SimdVectorStorage<typename Storage0::Scalar, Storage0::dims> {};
            } else {
                return GenericVectorStorage<typename Storage0::Scalar, Storage0::dims> {};
            }
        }

        template<typename T, size_t N>
        using VectorStorage = typename VectorStorageType<T, N>::type;

        template<typename Storage0, typename Storage1>
        using VectorStorageMerger = decltype(vectorStorageTypeMerger<Storage0, Storage1>());

        template<typename Derived>
        class VectorBase {
        public:
            using Scalar         = typename typetraits::TypeInfo<Derived>::Scalar;
            using IndexType      = typename typetraits::TypeInfo<Derived>::IndexType;
            using IndexTypeConst = typename typetraits::TypeInfo<Derived>::IndexTypeConst;
            using GetType        = typename typetraits::TypeInfo<Derived>::GetType;

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const auto &derived() const {
                return static_cast<const Derived &>(*this);
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto &derived() {
                return static_cast<Derived &>(*this);
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto eval() const { return derived(); }

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

            LIBRAPID_NODISCARD virtual std::string str(const std::string &format) const {
                return derived().str(format);
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE virtual GetType _get(size_t index) const {
                return derived()._get(index);
            }
        };
    } // namespace vectorDetail

    template<typename ScalarType, size_t NumDims>
    class Vector;

    namespace vectorDetail {
        template<typename LHS, typename RHS, typename Op>
        struct BinaryVecOp;

        template<typename Val, typename Op>
        struct UnaryVecOp;

        template<typename Scalar, size_t N, typename LHS, typename RHS, typename Op,
                 size_t... Indices>
        LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
                                               const BinaryVecOp<LHS, RHS, Op> &src,
                                               std::index_sequence<Indices...>);

        template<typename Scalar, size_t N, typename Val, typename Op, size_t... Indices>
        LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
                                               const UnaryVecOp<Val, Op> &src,
                                               std::index_sequence<Indices...>);

        template<typename Scalar, size_t N, typename LHS, typename RHS, typename Op>
        LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst,
                                           const BinaryVecOp<LHS, RHS, Op> &src);

        template<typename Scalar, size_t N, typename Val, typename Op>
        LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst, const UnaryVecOp<Val, Op> &src);
    } // namespace vectorDetail

    template<typename ScalarType, size_t NumDims>
    class Vector;
} // namespace librapid

#endif // LIBRAPID_MATH_VECTOR_FORWARD_HPP