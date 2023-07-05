#ifndef LIBRAPID_ARRAY_LINALG_ARRAY_MULTIPLY_HPP
#define LIBRAPID_ARRAY_LINALG_ARRAY_MULTIPLY_HPP

namespace librapid { namespace linalg {
	enum class MatmulClass {
		DOT,   // Vector-vector dot product
		GEMV,  // Matrix-vector product
		GEMM,  // Matrix-matrix product
		OUTER, // Outer product
	};

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha = typename StorageTypeA::Scalar,
			 typename Beta	= typename StorageTypeB::Scalar>
	class ArrayMultiply {
	public:
		using TypeA		= array::ArrayContainer<ShapeTypeA, StorageTypeA>;
		using TypeB		= array::ArrayContainer<ShapeTypeB, StorageTypeB>;
		using ScalarA	= typename StorageTypeA::Scalar;
		using ScalarB	= typename StorageTypeB::Scalar;
		using ShapeType = ShapeTypeA;
		using Backend	= typename typetraits::TypeInfo<TypeA>::Backend;
		using BackendB	= typename typetraits::TypeInfo<TypeB>::Backend;

		static_assert(std::is_same_v<Backend, BackendB>, "Backend of A and B must match");

		ArrayMultiply() = delete;

		ArrayMultiply(const ArrayMultiply &) = default;

		ArrayMultiply(ArrayMultiply &&) noexcept = default;

		ArrayMultiply(bool transA, bool transB, const TypeA &a, Alpha alpha, const TypeB &b,
					  Beta beta);

		ArrayMultiply &operator=(const ArrayMultiply &) = default;

		ArrayMultiply &operator=(ArrayMultiply &&) noexcept = default;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE MatmulClass matmulClass() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ScalarA alpha() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ScalarB beta() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool transA() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool transB() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const TypeA &a() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const TypeB &b() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE TypeA &a();
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE TypeB &b();

		template<typename StorageType>
		void applyTo(ArrayRef<StorageType> &out) const;

	private:
		bool m_transA;
		bool m_transB;
		TypeA m_a;
		ScalarA m_alpha;
		TypeB m_b;
		ScalarB m_beta;
	};

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::ArrayMultiply(
	  bool transA, bool transB, const TypeA &a, Alpha alpha, const TypeB &b, Beta beta) :
			m_transA(transA),
			m_transB(transB), m_a(a), m_alpha(static_cast<ScalarA>(alpha)), m_b(b),
			m_beta(static_cast<ScalarB>(beta)) {}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::matmulClass()
	  const -> MatmulClass {
		const auto &shapeA = m_a.shape();
		const auto &shapeB = m_b.shape();

		if (shapeA.ndim() == 1 && shapeB.ndim() == 1) {
			LIBRAPID_ASSERT(shapeA[0] == shapeB[0],
							"Vector dimensions must. Expected: {} -- Got: {}",
							shapeA[0],
							shapeB[0]);

			return MatmulClass::DOT;
		} else if (shapeA.ndim() == 1 && shapeB.ndim() == 2) {
			LIBRAPID_ASSERT(shapeA[0] == shapeB[int(!m_transB)],
							"Columns of OP(B) must match elements of A. Expected: {} -- Got: {}",
							shapeA[0],
							shapeB[int(!m_transB)]);

			return MatmulClass::GEMV;
		} else if (shapeA.ndim() == 2 && shapeB.ndim() == 1) {
			LIBRAPID_ASSERT(shapeA[int(m_transA)] == shapeB[0],
							"Rows of OP(A) must match elements of B. Expected: {} -- Got: {}",
							shapeA[int(m_transA)],
							shapeB[0]);

			return MatmulClass::GEMV;
		} else if (shapeA.ndim() == 2 && shapeB.ndim() == 2) {
			LIBRAPID_ASSERT(m_a.ndim() == 2,
							"First argument to gemm must be a matrix. Expected: 2 -- Got: {}",
							m_a.ndim());
			LIBRAPID_ASSERT(m_b.ndim() == 2,
							"Second argument to gemm must be a matrix. Expected: 2 -- Got: {}",
							m_b.ndim());
			LIBRAPID_ASSERT(m_a.shape()[int(!m_transA)] == m_b.shape()[int(m_transB)],
							"Inner dimensions of matrices must match. Expected: {} -- Got: {}",
							m_a.shape()[int(!m_transA)],
							m_b.shape()[int(m_transB)]);

			return MatmulClass::GEMM;
		} else {
			return MatmulClass::OUTER;
		}
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::shape() const
	  -> ShapeType {
		const auto &shapeA		= m_a.shape();
		const auto &shapeB		= m_b.shape();
		MatmulClass matmulClass = this->matmulClass();

		switch (matmulClass) {
			case MatmulClass::DOT: {
				return {1};
			}
			case MatmulClass::GEMV: {
				if (shapeA.ndim() == 1) {
					return {shapeA[0]};
				} else {
					return {shapeA[int(!m_transA)]};
				}
			}
			case MatmulClass::GEMM: {
				return {m_a.shape()[int(m_transA)], m_b.shape()[int(!m_transB)]};
			}
			case MatmulClass::OUTER: {
				LIBRAPID_NOT_IMPLEMENTED;
				return {1};
			}
		}
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::alpha() const
	  -> ScalarA {
		return m_alpha;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::beta() const
	  -> ScalarB {
		return m_beta;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	bool
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::transA() const {
		return m_transA;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	bool
	ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::transB() const {
		return m_transB;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::a() const
	  -> const TypeA & {
		return m_a;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::b() const
	  -> const TypeB & {
		return m_b;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::a()
	  -> TypeA & {
		return m_a;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::b()
	  -> TypeB & {
		return m_b;
	}

	template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
			 typename Alpha, typename Beta>
	template<typename StorageType>
	void ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::applyTo(
	  ArrayRef<StorageType> &out) const {
		LIBRAPID_ASSERT(out.shape() == shape(),
						"Output shape must match shape of gemm operation. Expected: {} -- Got: {}",
						shape(),
						out.shape());

		auto m = int64_t(m_a.shape()[m_transA]);
		auto n = int64_t(m_b.shape()[1 - m_transB]);
		auto k = int64_t(m_a.shape()[1 - m_transA]);

		auto lda = int64_t(m_a.shape()[1]);
		auto ldb = int64_t(m_b.shape()[1]);
		auto ldc = int64_t(out.shape()[1]);

		gemm(m_transA,
			 m_transB,
			 m,
			 n,
			 k,
			 m_alpha,
			 m_a.storage().data(),
			 lda,
			 m_beta,
			 m_b.storage().data(),
			 ldb,
			 out.storage().data(),
			 ldc,
			 Backend());
	}
}}	   // namespace librapid::linalg

#endif // LIBRAPID_ARRAY_LINALG_ARRAY_MULTIPLY_HPP