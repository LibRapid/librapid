#ifndef LIBRAPID_MATH_VECTOR_HPP
#define LIBRAPID_MATH_VECTOR_HPP

#include "../simd/simd.hpp" // Required for SIMD operations

namespace librapid {
    namespace typetraits {
        template<typename T>
        struct IsVector : std::false_type {};

        template<typename T, size_t N>
        struct IsVector<vectorDetail::GenericVectorStorage<T, N>> : std::true_type {};

        template<typename T, size_t N>
        struct IsVector<vectorDetail::SimdVectorStorage<T, N>> : std::true_type {};

        template<typename T, size_t N>
        struct IsVector<Vector<T, N>> : std::true_type {};

        template<typename LHS, typename RHS, typename Op>
        struct IsVector<vectorDetail::BinaryVecOp<LHS, RHS, Op>> : std::true_type {};

        template<typename VAL, typename Op>
        struct IsVector<vectorDetail::UnaryVecOp<VAL, Op>> : std::true_type {};

        template<typename T, size_t N>
        struct TypeInfo<vectorDetail::GenericVectorStorage<T, N>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Vector;
            using Scalar                               = T;
            using IndexType                            = T &;
            using IndexTypeConst                       = const T &;
            using GetType                              = const T &;

            using StorageType = vectorDetail::GenericVectorStorage<T, N>;

            static constexpr size_t length = N;
        };

        template<typename T, size_t N>
        struct TypeInfo<vectorDetail::SimdVectorStorage<T, N>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Vector;
            using Scalar                               = T;
            using Packet                               = typename TypeInfo<T>::Packet;
            using IndexType                            = Scalar &;
            using IndexTypeConst                       = const Scalar &;
            using GetType                              = const Packet &;

            using StorageType = vectorDetail::SimdVectorStorage<T, N>;

            static constexpr size_t packetWidth = TypeInfo<T>::packetWidth;
            static constexpr size_t length =
              (N + TypeInfo<T>::packetWidth - 1) / TypeInfo<T>::packetWidth;
        };

        template<typename ScalarType, size_t NumDims>
        struct TypeInfo<Vector<ScalarType, NumDims>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Vector;
            using Scalar                               = ScalarType;
            static constexpr size_t dims               = NumDims;
            using StorageType              = vectorDetail::VectorStorage<Scalar, NumDims>;
            static constexpr size_t length = StorageType::length;
            using IndexTypeConst           = typename StorageType::IndexTypeConst;
            using IndexType                = typename StorageType::IndexType;
            using GetType                  = typename StorageType::GetType;
        };

        template<typename LHS, typename RHS, typename Op>
        struct TypeInfo<vectorDetail::BinaryVecOp<LHS, RHS, Op>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Vector;
            using ScalarLHS                            = typename typetraits::TypeInfo<LHS>::Scalar;
            using ScalarRHS                            = typename typetraits::TypeInfo<RHS>::Scalar;
            using Scalar = decltype(Op()(std::declval<ScalarLHS>(), std::declval<ScalarRHS>()));
            using IndexTypeConst           = Scalar;
            using IndexType                = Scalar;
            using StorageType              = typename vectorDetail::VectorStorageMerger<LHS, RHS>;
            static constexpr size_t dims   = StorageType::dims;
            static constexpr size_t length = StorageType::length;
            using GetType                  = typename std::decay_t<typename StorageType::GetType>;
        };

        template<typename Val, typename Op>
        struct TypeInfo<vectorDetail::UnaryVecOp<Val, Op>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Vector;
            using Scalar                               = typename typetraits::TypeInfo<Val>::Scalar;
            using IndexTypeConst                       = Scalar;
            using IndexType                            = Scalar;
            using StorageType            = typename vectorDetail::VectorStorage<Scalar, Val::dims>;
            static constexpr size_t dims = StorageType::dims;
            static constexpr size_t length = StorageType::length;
            using GetType                  = typename std::decay_t<typename StorageType::GetType>;
        };
    } // namespace typetraits

    namespace vectorDetail {
        template<typename T, size_t N, typename... Args, size_t... Indices>
        void vectorStorageAssigner(std::index_sequence<Indices...>, GenericVectorStorage<T, N> &dst,
                                   const Args &...args) {
            ((dst[Indices] = args), ...);
        }

        template<typename T, size_t N, typename... Args, size_t... Indices>
        void vectorStorageAssigner(std::index_sequence<Indices...>, SimdVectorStorage<T, N> &dst,
                                   const Args &...args) {
            ((dst[Indices] = args), ...);
        }

        template<typename T, size_t N, typename T2, size_t N2, size_t... Indices>
        void vectorStorageAssigner(std::index_sequence<Indices...>, GenericVectorStorage<T, N> &dst,
                                   const GenericVectorStorage<T2, N2> &src) {
            ((dst[Indices] = src[Indices]), ...);
        }

        template<typename T, size_t N, typename T2, size_t N2, size_t... Indices>
        void vectorStorageAssigner(std::index_sequence<Indices...>, GenericVectorStorage<T, N> &dst,
                                   const SimdVectorStorage<T2, N2> &src) {
            ((dst[Indices] = src[Indices]), ...);
        }

        template<typename T, size_t N, typename T2, size_t N2, size_t... Indices>
        void vectorStorageAssigner(std::index_sequence<Indices...>, SimdVectorStorage<T, N> &dst,
                                   const GenericVectorStorage<T2, N2> &src) {
            ((dst[Indices] = src[Indices]), ...);
        }

        template<typename T, size_t N, typename T2, size_t N2, size_t... Indices>
        void vectorStorageAssigner(std::index_sequence<Indices...>, SimdVectorStorage<T, N> &dst,
                                   const SimdVectorStorage<T2, N2> &src) {
            ((dst[Indices] = src[Indices]), ...);
        }

        template<typename ScalarType, size_t NumDims>
        struct GenericVectorStorage {
            using Scalar                   = ScalarType;
            static constexpr size_t dims   = NumDims;
            static constexpr size_t length = typetraits::TypeInfo<GenericVectorStorage>::length;
            using IndexType = typename typetraits::TypeInfo<GenericVectorStorage>::IndexType;
            using IndexTypeConst =
              typename typetraits::TypeInfo<GenericVectorStorage>::IndexTypeConst;
            using GetType = typename typetraits::TypeInfo<GenericVectorStorage>::GetType;

            // Scalar data[length] {};
            // std::array<Scalar, length> data {};

#if defined(LIBRAPID_NATIVE_ARCH) && !defined(LIBRAPID_APPLE)
            alignas(LIBRAPID_DEFAULT_MEM_ALIGN) std::array<Scalar, length> data {};
#else
            // No memory alignment on Apple platforms or if it is disabled
            std::array<Scalar, length> data {};
#endif

            template<typename... Args>
            GenericVectorStorage(Args... args) : data {args...} {}

            template<typename T>
            GenericVectorStorage(const T &other) {
                for (size_t i = 0; i < length; ++i) { data[i] = other[i]; }
            }

            template<typename T>
            GenericVectorStorage(const std::initializer_list<T> &other) {
                LIBRAPID_ASSERT(other.size() <= dims,
                                "Initializer list for Vector is too long ({} > {})",
                                other.size(),
                                dims);
                const size_t minDims = (other.size() < dims) ? other.size() : dims;
                for (size_t i = 0; i < minDims; ++i) { this->operator[](i) = *(other.begin() + i); }
            }

            template<typename T>
            GenericVectorStorage(const std::vector<T> &other) {
                LIBRAPID_ASSERT(other.size() <= dims,
                                "Initializer list for Vector is too long ({} > {})",
                                other.size(),
                                dims);
                const size_t minDims = (other.size() < dims) ? other.size() : dims;
                for (size_t i = 0; i < minDims; ++i) { this->operator[](i) = other[i]; }
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst
            operator[](int64_t index) const {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                return data[index];
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                return data[index];
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar sum() const {
                Scalar sum = Scalar(0);
                for (size_t i = 0; i < dims; ++i) { sum += data[i]; }
                return sum;
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar sum2() const {
                Scalar sum = Scalar(0);
                for (size_t i = 0; i < dims; ++i) { sum += data[i] * data[i]; }
                return sum;
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const Scalar &_get(size_t index) const {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                return data[index];
            }

            LIBRAPID_ALWAYS_INLINE void _set(size_t index, const Scalar &value) {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                data[index] = value;
            }
        };

        template<typename ScalarType, size_t NumDims>
        struct SimdVectorStorage {
            using Scalar                        = ScalarType;
            static constexpr size_t dims        = NumDims;
            using Packet                        = typename typetraits::TypeInfo<Scalar>::Packet;
            static constexpr size_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;
            static constexpr size_t length      = (dims + packetWidth - 1) / packetWidth;

            using IndexType      = typename typetraits::TypeInfo<SimdVectorStorage>::IndexType;
            using IndexTypeConst = typename typetraits::TypeInfo<SimdVectorStorage>::IndexTypeConst;
            using GetType        = typename typetraits::TypeInfo<SimdVectorStorage>::GetType;

            static_assert(typetraits::TypeInfo<Scalar>::packetWidth > 1,
                          "SimdVectorStorage can only be used with SIMD types");

#if defined(LIBRAPID_NATIVE_ARCH) && !defined(LIBRAPID_APPLE)
            alignas(LIBRAPID_DEFAULT_MEM_ALIGN) std::array<Packet, length> data {};
#else
            // No memory alignment on Apple platforms or if it is disabled
            std::array<Packet, length> data {};
#endif

            template<typename... Args>
            explicit SimdVectorStorage(Args... args) {
                constexpr size_t minLength = (sizeof...(Args) < dims) ? sizeof...(Args) : dims;
                vectorDetail::vectorStorageAssigner(
                  std::make_index_sequence<minLength>(), *this, args...);
            }

            template<typename T>
            SimdVectorStorage(const std::initializer_list<T> &other) {
                LIBRAPID_ASSERT(other.size() <= dims,
                                "Initializer list for Vector is too long ({} > {})",
                                other.size(),
                                dims);
                const size_t minDims = (other.size() < dims) ? other.size() : dims;
                for (size_t i = 0; i < minDims; ++i) { this->operator[](i) = *(other.begin() + i); }
            }

            template<typename T>
            SimdVectorStorage(const std::vector<T> &other) {
                LIBRAPID_ASSERT(other.size() <= dims,
                                "Initializer list for Vector is too long ({} > {})",
                                other.size(),
                                dims);
                const size_t minDims = (other.size() < dims) ? other.size() : dims;
                for (size_t i = 0; i < minDims; ++i) { this->operator[](i) = other[i]; }
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst
            operator[](int64_t index) const {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                const int64_t packetIndex  = index / packetWidth;
                const int64_t elementIndex = index % packetWidth;
                return data[packetIndex].get(elementIndex);
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                const int64_t packetIndex  = index / packetWidth;
                const int64_t elementIndex = index % packetWidth;
                return data[packetIndex][elementIndex];
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto sum() const -> Scalar {
                Packet sum = Packet(0);
                for (size_t i = 0; i < length; ++i) { sum += data[i]; }
                return sum.sum();
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto sum2() const -> Scalar {
                Packet sum = Packet(0);
                for (size_t i = 0; i < length; ++i) { sum += data[i] * data[i]; }
                return sum.sum();
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const Packet &_get(size_t index) const {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                return data[index];
            }

            LIBRAPID_ALWAYS_INLINE void _set(size_t index, const Packet &value) {
                LIBRAPID_ASSERT(index >= 0 && index < dims,
                                "Index {} out of bounds for Vector of length {}",
                                index,
                                length);
                data[index] = value;
            }
        };

        template<typename T>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto
        scalarSubscriptHelper(const T &val, size_t index) {
            return val;
        }

        template<typename ScalarType, size_t NumDims>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto
        scalarSubscriptHelper(const Vector<ScalarType, NumDims> &val, size_t index) {
            return val[index];
        }

        template<typename LHS, typename RHS, typename Op>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto
        scalarSubscriptHelper(const BinaryVecOp<LHS, RHS, Op> &val, size_t index) {
            return val[index];
        }

        template<typename Val, typename Op>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto
        scalarSubscriptHelper(const UnaryVecOp<Val, Op> &val, size_t index) {
            return val[index];
        }

        template<typename T>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto scalarGetHelper(const T &val,
                                                                                 size_t index) {
            return val;
        }

        template<typename ScalarType, size_t NumDims>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto constexpr scalarGetHelper(
          const Vector<ScalarType, NumDims> &val, size_t index) {
            return val._get(index);
        }

        template<typename ScalarType, size_t NumDims>
        LIBRAPID_NODISCARD
          LIBRAPID_ALWAYS_INLINE auto constexpr scalarGetHelper(Vector<ScalarType, NumDims> &val,
                                                                size_t index) {
            return val._get(index);
        }

        template<typename LHS, typename RHS, typename Op>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto constexpr scalarGetHelper(
          const BinaryVecOp<LHS, RHS, Op> &val, size_t index) {
            return val._get(index);
        }

        template<typename Val, typename Op>
        LIBRAPID_NODISCARD
          LIBRAPID_ALWAYS_INLINE auto constexpr scalarGetHelper(const UnaryVecOp<Val, Op> &val,
                                                                size_t index) {
            return val._get(index);
        }

        template<typename T>
        struct VectorScalarStorageExtractor {
            using type = std::false_type;
        };

        template<typename ScalarType, size_t NumDims>
        struct VectorScalarStorageExtractor<Vector<ScalarType, NumDims>> {
            using type = typename typetraits::TypeInfo<Vector<ScalarType, NumDims>>::StorageType;
        };

        template<typename LHS, typename RHS, typename Op>
        struct VectorScalarStorageExtractor<BinaryVecOp<LHS, RHS, Op>> {
            using type = typename typetraits::TypeInfo<BinaryVecOp<LHS, RHS, Op>>::StorageType;
        };

        template<typename Val, typename Op>
        struct VectorScalarStorageExtractor<UnaryVecOp<Val, Op>> {
            using type = typename typetraits::TypeInfo<UnaryVecOp<Val, Op>>::StorageType;
        };

        template<typename T>
        struct VectorScalarDimensionExtractor {
            static constexpr size_t value = 0;
        };

        template<typename ScalarType, size_t NumDims>
        struct VectorScalarDimensionExtractor<Vector<ScalarType, NumDims>> {
            static constexpr size_t value = NumDims;
        };

        template<typename LHS, typename RHS, typename Op>
        struct VectorScalarDimensionExtractor<BinaryVecOp<LHS, RHS, Op>> {
            static constexpr size_t value = BinaryVecOp<LHS, RHS, Op>::dims;
        };

        template<typename Val, typename Op>
        struct VectorScalarDimensionExtractor<UnaryVecOp<Val, Op>> {
            static constexpr size_t value = UnaryVecOp<Val, Op>::dims;
        };
    } // namespace vectorDetail

    template<typename ScalarType, size_t NumDims>
    class Vector : public vectorDetail::VectorBase<Vector<ScalarType, NumDims>> {
    public:
        using Scalar                   = ScalarType;
        static constexpr size_t dims   = NumDims;
        using StorageType              = vectorDetail::VectorStorage<Scalar, NumDims>;
        static constexpr size_t length = StorageType::length;
        using IndexTypeConst           = typename StorageType::IndexTypeConst;
        using IndexType                = typename StorageType::IndexType;
        using GetType                  = typename StorageType::GetType;

        Vector()                        = default;
        Vector(const Vector &other)     = default;
        Vector(Vector &&other) noexcept = default;

        template<typename... Args>
        explicit Vector(Args... args) : m_data {args...} {}

        template<typename T>
        Vector(const std::initializer_list<T> &args) : m_data(args) {}

        template<typename T>
        explicit Vector(const std::vector<T> &args) : m_data(args) {}

        template<typename OtherScalar, size_t OtherDims>
        explicit Vector(const Vector<OtherScalar, OtherDims> &other) {
            *this = other.template cast<Scalar, dims>();
        }

        template<typename LHS, typename RHS, typename Op>
        explicit Vector(const vectorDetail::BinaryVecOp<LHS, RHS, Op> &other) {
            vectorDetail::assign(*this, other);
        }

        template<typename Val, typename Op>
        explicit Vector(const vectorDetail::UnaryVecOp<Val, Op> &other) {
            vectorDetail::assign(*this, other);
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto zero() -> Vector { return Vector(); }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto one() -> Vector {
            Vector ret;
            for (size_t i = 0; i < dims; ++i) { ret[i] = Scalar(1); }
            return ret;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto full(Scalar val) -> Vector {
            Vector ret;
            for (size_t i = 0; i < dims; ++i) { ret[i] = val; }
            return ret;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto random(Scalar lower = 0,
                                                                     Scalar upper = 1) {
            Vector ret;
            for (size_t i = 0; i < dims; ++i) { ret[i] = ::librapid::random<Scalar>(lower, upper); }
            return ret;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto
        random(const Vector &lower = Vector::zero(), const Vector &upper = Vector::one()) {
            Vector ret;
            for (size_t i = 0; i < dims; ++i) {
                ret[i] = ::librapid::random<Scalar>(lower[i], upper[i]);
            }
            return ret;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto fromPolar(Scalar r, Scalar theta) {
            return Vector(::librapid::cos(theta) * r, ::librapid::sin(theta) * r);
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE static auto fromPolar(Scalar r, Scalar theta,
                                                                        Scalar phi) {
            return Vector(::librapid::cos(theta) * ::librapid::cos(phi) * r,
                          ::librapid::sin(theta) * ::librapid::cos(phi) * r,
                          ::librapid::sin(phi) * r);
        }

        auto operator=(const Vector &other) -> Vector     & = default;
        auto operator=(Vector &&other) noexcept -> Vector & = default;

        template<typename OtherScalar, size_t OtherDims>
        auto operator=(const Vector<OtherScalar, OtherDims> &other) -> Vector & {
            *this = other.template cast<Scalar, dims>();
            return *this;
        }

        template<typename LHS, typename RHS, typename Op>
        auto operator=(const vectorDetail::BinaryVecOp<LHS, RHS, Op> &other) -> Vector & {
            vectorDetail::assign(*this, other);
            return *this;
        }

        template<typename Val, typename Op>
        auto operator=(const vectorDetail::UnaryVecOp<Val, Op> &other) -> Vector & {
            vectorDetail::assign(*this, other);
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst
        operator[](int64_t index) const override {
            return m_data[index];
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) override {
            return m_data[index];
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Vector eval() const { return *this; }

        template<typename NewScalar, size_t NewDims>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cast() const {
            using NewVectorType      = Vector<NewScalar, NewDims>;
            constexpr size_t minDims = (NewVectorType::dims < dims) ? NewVectorType::dims : dims;
            NewVectorType ret;
            vectorDetail::vectorStorageAssigner(
              std::make_index_sequence<minDims>(), ret.storage(), m_data);
            return ret;
        }

        template<typename NewScalar, size_t NewDims>
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator Vector<NewScalar, NewDims>() const {
            return cast<NewScalar, NewDims>();
        }

#define LIBRAPID_VECTOR_INPLACE_OP(OP_)                                                            \
    template<typename Other>                                                                       \
    LIBRAPID_ALWAYS_INLINE Vector &operator OP_##=(const Other &other) {                           \
        return *this = *this OP_ other;                                                            \
    }

        LIBRAPID_VECTOR_INPLACE_OP(+)
        LIBRAPID_VECTOR_INPLACE_OP(-)
        LIBRAPID_VECTOR_INPLACE_OP(*)
        LIBRAPID_VECTOR_INPLACE_OP(/)
        LIBRAPID_VECTOR_INPLACE_OP(%)
        LIBRAPID_VECTOR_INPLACE_OP(&)
        LIBRAPID_VECTOR_INPLACE_OP(|)
        LIBRAPID_VECTOR_INPLACE_OP(^)
        LIBRAPID_VECTOR_INPLACE_OP(<<)
        LIBRAPID_VECTOR_INPLACE_OP(>>)

        LIBRAPID_NODISCARD std::string str(const std::string &format) const override {
            std::string ret = "(";
            for (size_t i = 0; i < dims; ++i) {
                ret += fmt::format(format, m_data[i]);
                if (i != dims - 1) { ret += ", "; }
            }

            return ret + ")";
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const StorageType &storage() const {
            return m_data;
        }
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE StorageType &storage() { return m_data; }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE GetType _get(size_t index) const override {
            return m_data._get(index);
        }

        LIBRAPID_ALWAYS_INLINE void _set(size_t index, const GetType &value) {
            m_data._set(index, value);
        }

    private:
        StorageType m_data;
    };

    namespace vectorDetail {
        template<typename LHS, typename RHS, typename Op>
        struct BinaryVecOp : public VectorBase<BinaryVecOp<LHS, RHS, Op>> {
            using Scalar                   = typename typetraits::TypeInfo<BinaryVecOp>::Scalar;
            using StorageLHS               = typename VectorScalarStorageExtractor<LHS>::type;
            using StorageRHS               = typename VectorScalarStorageExtractor<RHS>::type;
            using StorageType              = VectorStorageMerger<StorageLHS, StorageRHS>;
            static constexpr size_t dims   = StorageType::dims;
            static constexpr size_t length = StorageType::length;
            using IndexTypeConst = typename typetraits::TypeInfo<BinaryVecOp>::IndexTypeConst;
            using IndexType      = typename typetraits::TypeInfo<BinaryVecOp>::IndexType;
            using GetType        = typename typetraits::TypeInfo<BinaryVecOp>::GetType;

            LHS left;
            RHS right;
            Op op;

            BinaryVecOp()                        = default;
            BinaryVecOp(const BinaryVecOp &)     = default;
            BinaryVecOp(BinaryVecOp &&) noexcept = default;

            BinaryVecOp(const LHS &lhs, const RHS &rhs, const Op &op) :
                    left(lhs), right(rhs), op(op) {}

            auto operator=(const BinaryVecOp &) -> BinaryVecOp     & = default;
            auto operator=(BinaryVecOp &&) noexcept -> BinaryVecOp & = default;

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Vector<Scalar, dims> eval() const {
                Vector<Scalar, dims> result(*this);
                return result;
            }

            template<typename NewScalar, size_t NewDims>
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cast() const {
                return eval().template cast<NewScalar, NewDims>();
            }

            template<typename NewScalar, size_t NewDims>
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator Vector<NewScalar, NewDims>() const {
                return cast<NewScalar, NewDims>();
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType
            operator[](int64_t index) const override {
                return op(scalarSubscriptHelper(left, index), scalarSubscriptHelper(right, index));
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) override {
                return op(scalarSubscriptHelper(left, index), scalarSubscriptHelper(right, index));
            }

            LIBRAPID_NODISCARD std::string str(const std::string &format) const override {
                return eval().str(format);
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE GetType _get(size_t index) const override {
                return op(scalarGetHelper(left, index), scalarGetHelper(right, index));
            }
        };

        template<typename Val, typename Op>
        struct UnaryVecOp : public VectorBase<UnaryVecOp<Val, Op>> {
            using Scalar                   = typename typetraits::TypeInfo<UnaryVecOp>::Scalar;
            using StorageType              = typename VectorScalarStorageExtractor<Val>::type;
            static constexpr size_t dims   = StorageType::dims;
            static constexpr size_t length = StorageType::length;
            using IndexTypeConst = typename typetraits::TypeInfo<UnaryVecOp>::IndexTypeConst;
            using IndexType      = typename typetraits::TypeInfo<UnaryVecOp>::IndexType;
            using GetType        = typename typetraits::TypeInfo<UnaryVecOp>::GetType;

            Val val;
            Op op;

            UnaryVecOp()                       = default;
            UnaryVecOp(const UnaryVecOp &)     = default;
            UnaryVecOp(UnaryVecOp &&) noexcept = default;

            UnaryVecOp(const Val &value, const Op &op) : val(value), op(op) {}

            auto operator=(const UnaryVecOp &) -> UnaryVecOp     & = default;
            auto operator=(UnaryVecOp &&) noexcept -> UnaryVecOp & = default;

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Vector<Scalar, dims> eval() const {
                Vector<Scalar, dims> result(*this);
                return result;
            }

            template<typename NewScalar, size_t NewDims>
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cast() const {
                return eval().template cast<NewScalar, NewDims>();
            }

            template<typename NewScalar, size_t NewDims>
            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator Vector<NewScalar, NewDims>() const {
                return cast<NewScalar, NewDims>();
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType
            operator[](int64_t index) const override {
                return op(scalarSubscriptHelper(val, index));
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) override {
                return op(scalarSubscriptHelper(val, index));
            }

            LIBRAPID_NODISCARD std::string str(const std::string &format) const override {
                return eval().str(format);
            }

            LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE GetType _get(size_t index) const override {
                return op(scalarGetHelper(val, index));
            }
        };

        template<typename Scalar, size_t N, typename LHS, typename RHS, typename Op,
                 size_t... Indices>
        LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
                                               const BinaryVecOp<LHS, RHS, Op> &src,
                                               std::index_sequence<Indices...>) {
            ((dst._set(
               Indices,
               src.op(scalarGetHelper(src.left, Indices), scalarGetHelper(src.right, Indices)))),
             ...);
        }

        template<typename Scalar, size_t N, typename Val, typename Op, size_t... Indices>
        LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
                                               const UnaryVecOp<Val, Op> &src,
                                               std::index_sequence<Indices...>) {
            ((dst._set(Indices, src.op(scalarGetHelper(src.val, Indices)))), ...);
        }

        template<typename Scalar, size_t N, typename LHS, typename RHS, typename Op>
        LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst,
                                           const BinaryVecOp<LHS, RHS, Op> &src) {
            using ScalarDst = typename typetraits::TypeInfo<Vector<Scalar, N>>::Scalar;
            using ScalarSrc = typename typetraits::TypeInfo<BinaryVecOp<LHS, RHS, Op>>::Scalar;
            if constexpr (std::is_same_v<ScalarDst, ScalarSrc>) {
                constexpr size_t lengthDst = Vector<Scalar, N>::length;
                constexpr size_t lengthSrc = BinaryVecOp<LHS, RHS, Op>::length;
                constexpr size_t minLength = (lengthDst < lengthSrc) ? lengthDst : lengthSrc;
                assignImpl(dst, src, std::make_index_sequence<minLength>());
            } else {
                dst = src.template cast<Scalar, N>();
            }
        }

        template<typename Scalar, size_t N, typename Val, typename Op>
        LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst, const UnaryVecOp<Val, Op> &src) {
            using ScalarDst = typename typetraits::TypeInfo<Vector<Scalar, N>>::Scalar;
            using ScalarSrc = typename typetraits::TypeInfo<UnaryVecOp<Val, Op>>::Scalar;
            if constexpr (std::is_same_v<ScalarDst, ScalarSrc>) {
                constexpr size_t lengthDst = Vector<Scalar, N>::length;
                constexpr size_t lengthSrc = UnaryVecOp<Val, Op>::length;
                constexpr size_t minLength = (lengthDst < lengthSrc) ? lengthDst : lengthSrc;
                assignImpl(dst, src, std::make_index_sequence<minLength>());
            } else {
                dst = src.template cast<Scalar, N>();
            }
        }

        template<typename T>
        constexpr auto scalarExtractor(const T &val) {
            return val;
        }

        // template<typename T, typename U>
        // constexpr auto scalarExtractor(const Vc_1::Detail::ElementReference<T, U> &val) {
        // 	using Scalar = typename Vc_1::Detail::ElementReference<T, U>::value_type;
        // 	return static_cast<Scalar>(val);
        // }

        template<typename NewScalar, size_t NewDims, typename T>
        constexpr auto scalarVectorCaster(const T &val) {
            return static_cast<NewScalar>(val);
        }

        template<typename NewScalar, size_t NewDims, typename ScalarType, size_t NumDims>
        constexpr auto scalarVectorCaster(const Vector<ScalarType, NumDims> &val) {
            return val.template cast<NewScalar, NewDims>();
        }

        template<typename NewScalar, size_t NewDims, typename LHS, typename RHS, typename Op>
        constexpr auto scalarVectorCaster(const BinaryVecOp<LHS, RHS, Op> &val) {
            return val.template cast<NewScalar, NewDims>();
        }

        template<typename NewScalar, size_t NewDims, typename Val, typename Op>
        constexpr auto scalarVectorCaster(const UnaryVecOp<Val, Op> &val) {
            return val.template cast<NewScalar, NewDims>();
        }

#define VECTOR_BINARY_OP(NAME_, OP_)                                                               \
    struct NAME_ {                                                                                 \
        template<typename A, typename B>                                                           \
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const A &a, const B &b) const {  \
            using namespace ::librapid::vectorDetail;                                              \
            return scalarExtractor(a) OP_ scalarExtractor(b);                                      \
        }                                                                                          \
    };                                                                                             \
                                                                                                   \
    template<typename LHS,                                                                         \
             typename RHS,                                                                         \
             typename std::enable_if_t<typetraits::IsVector<LHS>::value ||                         \
                                         typetraits::IsVector<RHS>::value,                         \
                                       int> = 0>                                                   \
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator OP_(const LHS &lhs, const RHS &rhs) {  \
        using namespace ::librapid::vectorDetail;                                                  \
        using ScalarLeft  = typename typetraits::TypeInfo<LHS>::Scalar;                            \
        using ScalarRight = typename typetraits::TypeInfo<RHS>::Scalar;                            \
        using Op          = NAME_;                                                                 \
                                                                                                   \
        if constexpr (std::is_same_v<ScalarLeft, ScalarRight>) {                                   \
            return BinaryVecOp<LHS, RHS, Op> {lhs, rhs, Op {}};                                    \
        } else {                                                                                   \
            using Scalar = decltype(std::declval<ScalarLeft>() + std::declval<ScalarRight>());     \
            constexpr size_t dimsLhs = VectorScalarDimensionExtractor<LHS>::value;                 \
            constexpr size_t dimsRhs = VectorScalarDimensionExtractor<RHS>::value;                 \
            constexpr size_t maxDims = (dimsLhs > dimsRhs) ? dimsLhs : dimsRhs;                    \
            return BinaryVecOp {scalarVectorCaster<Scalar, maxDims>(lhs),                          \
                                scalarVectorCaster<Scalar, maxDims>(rhs),                          \
                                Op {}};                                                            \
        }                                                                                          \
    }

#define VECTOR_UNARY_OP(NAME_, OP_NAME_, OP_)                                                      \
    struct NAME_ {                                                                                 \
        template<typename Val,                                                                     \
                 typename std::enable_if_t<typetraits::IsVector<Val>::value, int> = 0>             \
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const Val &val) const {          \
            using namespace ::librapid::vectorDetail;                                              \
            return OP_(scalarExtractor(val));                                                      \
        }                                                                                          \
    };                                                                                             \
                                                                                                   \
    template<typename Val, typename std::enable_if_t<typetraits::IsVector<Val>::value, int> = 0>   \
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto OP_NAME_(const Val &val) {                      \
        using namespace ::librapid::vectorDetail;                                                  \
        using Op = NAME_;                                                                          \
        return ::librapid::vectorDetail::UnaryVecOp<Val, Op> {val, Op {}};                         \
    }

        VECTOR_BINARY_OP(Add, +);
        VECTOR_BINARY_OP(Sub, -);
        VECTOR_BINARY_OP(Mul, *);
        VECTOR_BINARY_OP(Div, /);
        VECTOR_BINARY_OP(Mod, %);
        VECTOR_BINARY_OP(BitAnd, &);
        VECTOR_BINARY_OP(BitOr, |);
        VECTOR_BINARY_OP(BitXor, ^);
        VECTOR_BINARY_OP(LeftShift, <<);
        VECTOR_BINARY_OP(RightShift, >>);
        VECTOR_BINARY_OP(And, &&);
        VECTOR_BINARY_OP(Or, ||);
        VECTOR_BINARY_OP(LessThan, <);
        VECTOR_BINARY_OP(GreaterThan, >);
        VECTOR_BINARY_OP(LessThanEqual, <=);
        VECTOR_BINARY_OP(GreaterThanEqual, >=);
        VECTOR_BINARY_OP(Equal, ==);
        VECTOR_BINARY_OP(NotEqual, !=);

        VECTOR_UNARY_OP(Not, operator!, !);
        VECTOR_UNARY_OP(BitNot, operator~, ~);
        VECTOR_UNARY_OP(Negate, operator-, -);
        VECTOR_UNARY_OP(Plus, operator+, +);

#define VECTOR_FUNC_STRUCT_DEF(NAME_)                                                              \
    struct Vector_##NAME_ {                                                                        \
        template<typename Val>                                                                     \
        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const Val &val) const {          \
            return ::librapid::NAME_(scalarExtractor(val));                                        \
        }                                                                                          \
    }

        VECTOR_FUNC_STRUCT_DEF(sin);
        VECTOR_FUNC_STRUCT_DEF(cos);
        VECTOR_FUNC_STRUCT_DEF(tan);
        VECTOR_FUNC_STRUCT_DEF(asin);
        VECTOR_FUNC_STRUCT_DEF(acos);
        VECTOR_FUNC_STRUCT_DEF(atan);
        VECTOR_FUNC_STRUCT_DEF(sinh);
        VECTOR_FUNC_STRUCT_DEF(cosh);
        VECTOR_FUNC_STRUCT_DEF(tanh);
        VECTOR_FUNC_STRUCT_DEF(asinh);
        VECTOR_FUNC_STRUCT_DEF(acosh);
        VECTOR_FUNC_STRUCT_DEF(atanh);
        VECTOR_FUNC_STRUCT_DEF(exp);
        VECTOR_FUNC_STRUCT_DEF(exp2);
        VECTOR_FUNC_STRUCT_DEF(exp10);
        VECTOR_FUNC_STRUCT_DEF(log);
        VECTOR_FUNC_STRUCT_DEF(log2);
        VECTOR_FUNC_STRUCT_DEF(log10);
        VECTOR_FUNC_STRUCT_DEF(sqrt);
        VECTOR_FUNC_STRUCT_DEF(cbrt);
    } // namespace vectorDetail

#define VECTOR_FUNC_IMPL_DEF(NAME_)                                                                \
    template<typename ScalarType, size_t NumDims>                                                  \
    auto NAME_(const Vector<ScalarType, NumDims> &vec) {                                           \
        return vectorDetail::UnaryVecOp {vec, vectorDetail::Vector_##NAME_ {}};                    \
    }                                                                                              \
                                                                                                   \
    template<typename LHS, typename RHS, typename Op>                                              \
    auto NAME_(const vectorDetail::BinaryVecOp<LHS, RHS, Op> &vec) {                               \
        return vectorDetail::UnaryVecOp {vec, vectorDetail::Vector_##NAME_ {}};                    \
    }                                                                                              \
                                                                                                   \
    template<typename Val, typename Op>                                                            \
    auto NAME_(const vectorDetail::UnaryVecOp<Val, Op> &vec) {                                     \
        return vectorDetail::UnaryVecOp {vec, vectorDetail::Vector_##NAME_ {}};                    \
    }

    VECTOR_FUNC_IMPL_DEF(sin)
    VECTOR_FUNC_IMPL_DEF(cos)
    VECTOR_FUNC_IMPL_DEF(tan)
    VECTOR_FUNC_IMPL_DEF(asin)
    VECTOR_FUNC_IMPL_DEF(acos)
    VECTOR_FUNC_IMPL_DEF(atan)
    VECTOR_FUNC_IMPL_DEF(sinh)
    VECTOR_FUNC_IMPL_DEF(cosh)
    VECTOR_FUNC_IMPL_DEF(tanh)
    VECTOR_FUNC_IMPL_DEF(asinh)
    VECTOR_FUNC_IMPL_DEF(acosh)
    VECTOR_FUNC_IMPL_DEF(atanh)
    VECTOR_FUNC_IMPL_DEF(exp)
    VECTOR_FUNC_IMPL_DEF(exp2)
    VECTOR_FUNC_IMPL_DEF(exp10)
    VECTOR_FUNC_IMPL_DEF(log)
    VECTOR_FUNC_IMPL_DEF(log2)
    VECTOR_FUNC_IMPL_DEF(log10)
    VECTOR_FUNC_IMPL_DEF(sqrt)
    VECTOR_FUNC_IMPL_DEF(cbrt)

    template<typename T, typename std::enable_if_t<typetraits::IsVector<T>::value, int> = 0>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto mag2(const T &val) {
        return val.eval().storage().sum2();
    }

    template<typename T, typename std::enable_if_t<typetraits::IsVector<T>::value, int> = 0>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto mag(const T &val) {
        return ::librapid::sqrt(mag2(val));
    }

    template<typename T, typename std::enable_if_t<typetraits::IsVector<T>::value, int> = 0>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto sum(const T &val) {
        return val.eval().storage().sum();
    }

    template<typename First, typename Second,
             typename std::enable_if_t<
               typetraits::IsVector<First>::value && typetraits::IsVector<Second>::value, int> = 0>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto dot(const First &first, const Second &second) {
        return (first * second).eval().storage().sum();
    }

    template<typename First, typename Second,
             typename std::enable_if_t<
               typetraits::IsVector<First>::value && typetraits::IsVector<Second>::value, int> = 0>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cross(const First &first, const Second &second) {
        LIBRAPID_ASSERT(typetraits::TypeInfo<First>::dims == 3 &&
                          typetraits::TypeInfo<Second>::dims == 3,
                        "Cross product is only defined for 3D vectors");
        using ScalarFirst  = typename typetraits::TypeInfo<First>::Scalar;
        using ScalarSecond = typename typetraits::TypeInfo<Second>::Scalar;
        using Scalar       = decltype(std::declval<ScalarFirst>() * std::declval<ScalarSecond>());

        Scalar x1 = first[0];
        Scalar y1 = first[1];
        Scalar z1 = first[2];
        Scalar x2 = second[0];
        Scalar y2 = second[1];
        Scalar z2 = second[2];

        return Vector<Scalar, 3> {y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2};
    }

    template<typename T, typename std::enable_if_t<typetraits::IsVector<T>::value, int> = 0>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto norm(const T &val) {
        return val / mag(val);
    }
} // namespace librapid

LIBRAPID_SIMPLE_IO_IMPL(typename Derived, librapid::vectorDetail::VectorBase<Derived>);
LIBRAPID_SIMPLE_IO_IMPL(typename T COMMA size_t N, librapid::Vector<T COMMA N>);
LIBRAPID_SIMPLE_IO_IMPL(typename LHS COMMA typename RHS COMMA typename Op,
                        librapid::vectorDetail::BinaryVecOp<LHS COMMA RHS COMMA Op>);
LIBRAPID_SIMPLE_IO_IMPL(typename Val COMMA typename Op,
                        librapid::vectorDetail::UnaryVecOp<Val COMMA Op>);

#endif // LIBRAPID_MATH_VECTOR_HPP
