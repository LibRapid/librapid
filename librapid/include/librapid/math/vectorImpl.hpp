#ifndef LIBRAPID_MATH_VECTOR_HPP
#define LIBRAPID_MATH_VECTOR_HPP

namespace librapid {
	namespace typetraits {
		template<typename T, uint64_t N>
		struct TypeInfo<vectorDetail::GenericVectorStorage<T, N>> {
			using Scalar		 = T;
			using IndexType		 = T &;
			using IndexTypeConst = const T &;
			using GetType		 = const T &;

			using StorageType = vectorDetail::GenericVectorStorage<T, N>;

			static constexpr uint64_t length = N;
		};

		template<typename T, uint64_t N>
		struct TypeInfo<vectorDetail::SimdVectorStorage<T, N>> {
			using Scalar		 = T;
			using Packet		 = typename TypeInfo<T>::Packet;
			using IndexType		 = typename std::decay_t<decltype(std::declval<Packet>()[0])>;
			using IndexTypeConst = typename std::decay_t<decltype(std::declval<const Packet>()[0])>;
			using GetType		 = const Packet &;

			using StorageType = vectorDetail::SimdVectorStorage<T, N>;

			static constexpr uint64_t packetWidth = TypeInfo<T>::packetWidth;
			static constexpr uint64_t length =
			  (N + TypeInfo<T>::packetWidth - 1) / TypeInfo<T>::packetWidth;
		};

		template<typename ScalarType, uint64_t NumDims>
		struct TypeInfo<Vector<ScalarType, NumDims>> {
			using Scalar					 = ScalarType;
			static constexpr uint64_t dims	 = NumDims;
			using StorageType				 = vectorDetail::VectorStorage<Scalar, NumDims>;
			static constexpr uint64_t length = StorageType::length;
			using IndexTypeConst			 = typename StorageType::IndexTypeConst;
			using IndexType					 = typename StorageType::IndexType;
			using GetType					 = typename StorageType::GetType;
			using StorageType				 = vectorDetail::VectorStorage<Scalar, NumDims>;
		};

		template<typename LHS, typename RHS, typename Op>
		struct TypeInfo<vectorDetail::BinaryVecOp<LHS, RHS, Op>> {
			using ScalarLHS = typename typetraits::TypeInfo<LHS>::Scalar;
			using ScalarRHS = typename typetraits::TypeInfo<RHS>::Scalar;
			using Scalar	= decltype(Op()(std::declval<ScalarLHS>(), std::declval<ScalarRHS>()));
			using IndexTypeConst			 = Scalar;
			using IndexType					 = Scalar;
			using StorageType				 = typename vectorDetail::VectorStorageMerger<LHS, RHS>;
			static constexpr uint64_t dims	 = StorageType::dims;
			static constexpr uint64_t length = StorageType::length;
			using GetType					 = typename StorageType::GetType;
		};
	} // namespace typetraits

	namespace vectorDetail {
		template<typename ScalarType, uint64_t NumDims>
		struct GenericVectorStorage {
			using Scalar					 = ScalarType;
			static constexpr uint64_t dims	 = NumDims;
			static constexpr uint64_t length = typetraits::TypeInfo<GenericVectorStorage>::length;
			using IndexType = typename typetraits::TypeInfo<GenericVectorStorage>::IndexType;
			using IndexTypeConst =
			  typename typetraits::TypeInfo<GenericVectorStorage>::IndexTypeConst;
			using GetType = typename typetraits::TypeInfo<GenericVectorStorage>::GetType;

			Scalar data[length] {};

			template<typename... Args>
			GenericVectorStorage(Args... args) : data {args...} {}

			template<typename T>
			GenericVectorStorage(const T &other) {
				for (uint64_t i = 0; i < length; ++i) { data[i] = other[i]; }
			}

			template<typename T>
			GenericVectorStorage(const std::initializer_list<T> &other) {
				LIBRAPID_ASSERT(other.size() <= dims,
								"Initializer list for Vector is too long ({} > {})",
								other.size(),
								dims);
				const uint64_t minDims = (other.size() < dims) ? other.size() : dims;
				for (uint64_t i = 0; i < minDims; ++i) { data[i] = *(other.begin() + i); }
			}

			template<typename T>
			GenericVectorStorage(const std::vector<T> &other) {
				LIBRAPID_ASSERT(other.size() <= dims,
								"Initializer list for Vector is too long ({} > {})",
								other.size(),
								dims);
				const uint64_t minDims = (other.size() < dims) ? other.size() : dims;
				for (uint64_t i = 0; i < minDims; ++i) { data[i] = other[i]; }
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst
			operator[](int64_t index) const {
				return data[index];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) {
				return data[index];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const Scalar &_get(uint64_t index) const {
				return data[index];
			}

			LIBRAPID_ALWAYS_INLINE void _set(uint64_t index, const Scalar &value) {
				data[index] = value;
			}
		};

		template<typename ScalarType, uint64_t NumDims>
		struct SimdVectorStorage {
			using Scalar						  = ScalarType;
			static constexpr uint64_t dims		  = NumDims;
			using Packet						  = typename typetraits::TypeInfo<Scalar>::Packet;
			static constexpr uint64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;
			static constexpr uint64_t length	  = (dims + packetWidth - 1) / packetWidth;

			using IndexType		 = typename typetraits::TypeInfo<SimdVectorStorage>::IndexType;
			using IndexTypeConst = typename typetraits::TypeInfo<SimdVectorStorage>::IndexTypeConst;
			using GetType		 = typename typetraits::TypeInfo<SimdVectorStorage>::GetType;

			static_assert(typetraits::TypeInfo<Scalar>::packetWidth > 1,
						  "SimdVectorStorage can only be used with SIMD types");

			Packet data[length] {};

			template<typename... Args>
			explicit SimdVectorStorage(Args... args) {
				constexpr uint64_t minLength = (sizeof...(Args) < dims) ? sizeof...(Args) : dims;
				vectorDetail::vectorStorageAssigner(
				  std::make_index_sequence<minLength>(), *this, args...);
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexTypeConst
			operator[](int64_t index) const {
				const int64_t packetIndex  = index / packetWidth;
				const int64_t elementIndex = index % packetWidth;
				return data[packetIndex][elementIndex];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) {
				const int64_t packetIndex  = index / packetWidth;
				const int64_t elementIndex = index % packetWidth;
				return data[packetIndex][elementIndex];
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const Packet &_get(uint64_t index) const {
				return data[index];
			}

			LIBRAPID_ALWAYS_INLINE void _set(uint64_t index, const Packet &value) {
				data[index] = value;
			}
		};

		template<typename T, uint64_t N, typename... Args, uint64_t... Indices>
		void vectorStorageAssigner(std::index_sequence<Indices...>, GenericVectorStorage<T, N> &dst,
								   const Args &...args) {
			((dst[Indices] = args), ...);
		}

		template<typename T, uint64_t N, typename... Args, uint64_t... Indices>
		void vectorStorageAssigner(std::index_sequence<Indices...>, SimdVectorStorage<T, N> &dst,
								   const Args &...args) {
			((dst[Indices] = args), ...);
		}

		template<typename T, uint64_t N, typename T2, uint64_t N2, uint64_t... Indices>
		void vectorStorageAssigner(std::index_sequence<Indices...>, GenericVectorStorage<T, N> &dst,
								   const GenericVectorStorage<T2, N2> &src) {
			((dst[Indices] = src[Indices]), ...);
		}

		template<typename T, uint64_t N, typename T2, uint64_t N2, uint64_t... Indices>
		void vectorStorageAssigner(std::index_sequence<Indices...>, GenericVectorStorage<T, N> &dst,
								   const SimdVectorStorage<T2, N2> &src) {
			((dst[Indices] = src[Indices]), ...);
		}

		template<typename T, uint64_t N, typename T2, uint64_t N2, uint64_t... Indices>
		void vectorStorageAssigner(std::index_sequence<Indices...>, SimdVectorStorage<T, N> &dst,
								   const GenericVectorStorage<T2, N2> &src) {
			((dst[Indices] = src[Indices]), ...);
		}

		template<typename T, uint64_t N, typename T2, uint64_t N2, uint64_t... Indices>
		void vectorStorageAssigner(std::index_sequence<Indices...>, SimdVectorStorage<T, N> &dst,
								   const SimdVectorStorage<T2, N2> &src) {
			((dst[Indices] = src[Indices]), ...);
		}

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		scalarSubscriptHelper(const T &val, uint64_t index) {
			return val;
		}

		template<typename ScalarType, uint64_t NumDims>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		scalarSubscriptHelper(const Vector<ScalarType, NumDims> &val, uint64_t index) {
			return val[index];
		}

		template<typename LHS, typename RHS, typename Op>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		scalarSubscriptHelper(const BinaryVecOp<LHS, RHS, Op> &val, uint64_t index) {
			return val[index];
		}

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalarGetHelper(const T &val,
																				 uint64_t index) {
			return val;
		}

		template<typename ScalarType, uint64_t NumDims>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		scalarGetHelper(const Vector<ScalarType, NumDims> &val, uint64_t index) {
			return val._get(index);
		}

		template<typename ScalarType, uint64_t NumDims>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		scalarGetHelper(Vector<ScalarType, NumDims> &val, uint64_t index) {
			return val._get(index);
		}

		template<typename LHS, typename RHS, typename Op>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		scalarGetHelper(const BinaryVecOp<LHS, RHS, Op> &val, uint64_t index) {
			return val._get(index);
		}

		template<typename T>
		struct VectorScalarStorageExtractor {
			using type = std::false_type;
		};

		template<typename ScalarType, uint64_t NumDims>
		struct VectorScalarStorageExtractor<Vector<ScalarType, NumDims>> {
			using type = typename typetraits::TypeInfo<Vector<ScalarType, NumDims>>::StorageType;
		};

		template<typename LHS, typename RHS, typename Op>
		struct VectorScalarStorageExtractor<BinaryVecOp<LHS, RHS, Op>> {
			using type = typename typetraits::TypeInfo<BinaryVecOp<LHS, RHS, Op>>::StorageType;
		};

		template<typename T>
		struct VectorScalarDimensionExtractor {
			static constexpr uint64_t value = 0;
		};

		template<typename ScalarType, uint64_t NumDims>
		struct VectorScalarDimensionExtractor<Vector<ScalarType, NumDims>> {
			static constexpr uint64_t value = NumDims;
		};

		template<typename LHS, typename RHS, typename Op>
		struct VectorScalarDimensionExtractor<BinaryVecOp<LHS, RHS, Op>> {
			static constexpr uint64_t value = BinaryVecOp<LHS, RHS, Op>::length;
		};
	} // namespace vectorDetail

	template<typename ScalarType, uint64_t NumDims>
	class Vector : public vectorDetail::VectorBase<Vector<ScalarType, NumDims>> {
	public:
		using Scalar					 = ScalarType;
		static constexpr uint64_t dims	 = NumDims;
		using StorageType				 = vectorDetail::VectorStorage<Scalar, NumDims>;
		static constexpr uint64_t length = StorageType::length;
		using IndexTypeConst			 = typename StorageType::IndexTypeConst;
		using IndexType					 = typename StorageType::IndexType;
		using GetType					 = typename StorageType::GetType;

		Vector()						= default;
		Vector(const Vector &other)		= default;
		Vector(Vector &&other) noexcept = default;

		template<typename... Args>
		explicit Vector(Args... args) : m_data {args...} {}

		template<typename OtherScalar, uint64_t OtherDims>
		explicit Vector(const Vector<OtherScalar, OtherDims> &other) {
			*this = other.template cast<Scalar, dims>();
		}

		template<typename LHS, typename RHS, typename Op>
		explicit Vector(const vectorDetail::BinaryVecOp<LHS, RHS, Op> &other) {
			vectorDetail::assign(*this, other);
		}

		auto operator=(const Vector &other) -> Vector	  & = default;
		auto operator=(Vector &&other) noexcept -> Vector & = default;

		template<typename LHS, typename RHS, typename Op>
		auto operator=(const vectorDetail::BinaryVecOp<LHS, RHS, Op> &other) -> Vector & {
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

		template<typename NewScalar, uint64_t NewDims>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cast() const {
			using NewVectorType		   = Vector<NewScalar, NewDims>;
			constexpr uint64_t minDims = (NewVectorType::dims < dims) ? NewVectorType::dims : dims;
			NewVectorType ret;
			vectorDetail::vectorStorageAssigner(
			  std::make_index_sequence<minDims>(), ret.storage(), m_data);
			return ret;
		}

		template<typename NewScalar, uint64_t NewDims>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator Vector<NewScalar, NewDims>() const {
			return cast<NewScalar, NewDims>();
		}

		LIBRAPID_NODISCARD std::string str(const std::string &format) const override {
			std::string ret = "(";
			for (uint64_t i = 0; i < dims; ++i) {
				ret += fmt::format(format, m_data[i]);
				if (i != dims - 1) { ret += ", "; }
			}

			return ret + ")";
		}

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const StorageType &storage() const {
			return m_data;
		}
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE StorageType &storage() { return m_data; }

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE GetType _get(uint64_t index) const override {
			return m_data._get(index);
		}

		LIBRAPID_ALWAYS_INLINE void _set(uint64_t index, const GetType &value) {
			m_data._set(index, value);
		}

	private:
		StorageType m_data;
	};

	namespace vectorDetail {
		template<typename LHS, typename RHS, typename Op>
		struct BinaryVecOp : public VectorBase<BinaryVecOp<LHS, RHS, Op>> {
			using Scalar = typename typetraits::TypeInfo<BinaryVecOp>::Scalar;
			// using StorageLHS				 = typename typetraits::TypeInfo<LHS>::StorageType;
			// using StorageRHS				 = typename typetraits::TypeInfo<RHS>::StorageType;

			using StorageLHS = typename VectorScalarStorageExtractor<LHS>::type;
			using StorageRHS = typename VectorScalarStorageExtractor<RHS>::type;

			using StorageType				 = VectorStorageMerger<StorageLHS, StorageRHS>;
			static constexpr uint64_t dims	 = StorageType::dims;
			static constexpr uint64_t length = StorageType::length;
			using IndexTypeConst = typename typetraits::TypeInfo<BinaryVecOp>::IndexTypeConst;
			using IndexType		 = typename typetraits::TypeInfo<BinaryVecOp>::IndexType;
			using GetType		 = typename typetraits::TypeInfo<BinaryVecOp>::GetType;

			LHS left;
			RHS right;
			Op op;

			BinaryVecOp()						 = default;
			BinaryVecOp(const BinaryVecOp &)	 = default;
			BinaryVecOp(BinaryVecOp &&) noexcept = default;

			BinaryVecOp(const LHS &lhs, const RHS &rhs, const Op &op) :
					left(lhs), right(rhs), op(op) {}

			auto operator=(const BinaryVecOp &) -> BinaryVecOp	   & = default;
			auto operator=(BinaryVecOp &&) noexcept -> BinaryVecOp & = default;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Vector<Scalar, dims> eval() const {
				Vector<Scalar, dims> result(*this);
				return result;
			}

			template<typename NewScalar, uint64_t NewDims>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cast() const {
				return eval().template cast<NewScalar, NewDims>();
			}

			template<typename NewScalar, uint64_t NewDims>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator Vector<NewScalar, NewDims>() const {
				return cast<NewScalar, NewDims>();
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType
			operator[](int64_t index) const override {
				// return op(left[index], right[index]);
				return op(scalarSubscriptHelper(left, index), scalarSubscriptHelper(right, index));
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE IndexType operator[](int64_t index) override {
				// return op(left[index], right[index]);
				return op(scalarSubscriptHelper(left, index), scalarSubscriptHelper(right, index));
			}

			LIBRAPID_NODISCARD std::string str(const std::string &format) const override {
				return eval().str(format);
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE GetType _get(uint64_t index) const override {
				// return op(left._get(index), right._get(index));
				return op(scalarGetHelper(left, index), scalarGetHelper(right, index));
			}
		};

		template<typename Scalar, uint64_t N, typename LHS, typename RHS, typename Op,
				 uint64_t... Indices>
		LIBRAPID_ALWAYS_INLINE void assignImpl(Vector<Scalar, N> &dst,
											   const BinaryVecOp<LHS, RHS, Op> &src,
											   std::index_sequence<Indices...>) {
			((dst._set(
			   Indices,
			   src.op(scalarGetHelper(src.left, Indices), scalarGetHelper(src.right, Indices)))),
			 ...);
		}

		template<typename Scalar, uint64_t N, typename LHS, typename RHS, typename Op>
		LIBRAPID_ALWAYS_INLINE void assign(Vector<Scalar, N> &dst,
										   const BinaryVecOp<LHS, RHS, Op> &src) {
			constexpr uint64_t lengthDst = Vector<Scalar, N>::length;
			constexpr uint64_t lengthSrc = BinaryVecOp<LHS, RHS, Op>::length;
			constexpr uint64_t minLength = (lengthDst < lengthSrc) ? lengthDst : lengthSrc;
			assignImpl(dst, src, std::make_index_sequence<minLength>());
		}

		template<typename T>
		auto scalarExtractor(const T &val) {
			return val;
		}

		template<typename T, typename U>
		auto scalarExtractor(const Vc_1::Detail::ElementReference<T, U> &val) {
			using Scalar = typename Vc_1::Detail::ElementReference<T, U>::value_type;
			return static_cast<Scalar>(val);
		}

		template<typename NewScalar, uint64_t NewDims, typename T>
		auto scalarVectorCaster(const T &val) {
			return val;
		}

		template<typename NewScalar, uint64_t NewDims, typename ScalarType, uint64_t NumDims>
		auto scalarVectorCaster(const Vector<ScalarType, NumDims> &val) {
			return val.template cast<NewScalar, NewDims>();
		}

		template<typename NewScalar, uint64_t NewDims, typename LHS, typename RHS, typename Op>
		auto scalarVectorCaster(const BinaryVecOp<LHS, RHS, Op> &val) {
			return val.template cast<NewScalar, NewDims>();
		}

#define VECTOR_BINARY_OP(NAME_, OP_)                                                               \
	struct NAME_ {                                                                                 \
		template<typename A, typename B>                                                           \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const A &a, const B &b) const {  \
			return scalarExtractor(a) OP_ scalarExtractor(b);                                      \
		}                                                                                          \
	};                                                                                             \
                                                                                                   \
	template<typename LHS, typename RHS>                                                           \
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator OP_(const LHS &lhs, const RHS &rhs) {  \
		using ScalarLeft  = typename typetraits::TypeInfo<LHS>::Scalar;                            \
		using ScalarRight = typename typetraits::TypeInfo<RHS>::Scalar;                            \
		using Op		  = NAME_;                                                                 \
                                                                                                   \
		if constexpr (std::is_same_v<ScalarLeft, ScalarRight>) {                                   \
			return BinaryVecOp<LHS, RHS, Op> {lhs, rhs, Op {}};                                    \
		} else {                                                                                   \
			using Scalar = decltype(std::declval<ScalarLeft>() + std::declval<ScalarRight>());     \
			constexpr uint64_t dimsLhs = VectorScalarDimensionExtractor<LHS>::value;               \
			constexpr uint64_t dimsRhs = VectorScalarDimensionExtractor<RHS>::value;               \
			constexpr uint64_t maxDims = (dimsLhs > dimsRhs) ? dimsLhs : dimsRhs;                  \
			return BinaryVecOp {scalarVectorCaster<Scalar, maxDims>(lhs),                          \
								scalarVectorCaster<Scalar, maxDims>(rhs),                          \
								Op {}};                                                            \
		}                                                                                          \
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
	} // namespace vectorDetail
} // namespace librapid

LIBRAPID_SIMPLE_IO_IMPL(typename Derived, librapid::vectorDetail::VectorBase<Derived>);
LIBRAPID_SIMPLE_IO_IMPL(typename T COMMA uint64_t N, librapid::Vector<T COMMA N>);
LIBRAPID_SIMPLE_IO_IMPL(typename LHS COMMA typename RHS COMMA typename Op,
						librapid::vectorDetail::BinaryVecOp<LHS COMMA RHS COMMA Op>);

#endif // LIBRAPID_MATH_VECTOR_HPP
