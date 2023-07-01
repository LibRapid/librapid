#ifndef LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP
#define LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP

namespace librapid { namespace linalg {
	template<typename Backend, typename Int, typename A, typename Alpha, typename B, typename Beta,
			 typename C>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, A *a, Int lda, Beta beta,
			  B *b, Int ldb, C *c, Int ldc, backend::CPU) {
		cxxblas::gemm(cxxblas::StorageOrder::RowMajor,
					  (transA ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
					  (transB ? cxxblas::Transpose::Trans : cxxblas::Transpose::NoTrans),
					  m,
					  n,
					  k,
					  alpha,
					  a,
					  lda,
					  b,
					  ldb,
					  beta,
					  c,
					  ldc);
	}

#if defined(LIBRAPID_HAS_OPENCL)

	template<typename Backend, typename Int, typename Alpha, typename Beta>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, cl::Buffer a, Int lda,
			  Beta beta, cl::Buffer b, Int ldb, cl::Buffer c, Int ldc, backend::OpenCL) {
		auto status = clblast::Gemm(clblast::Layout::kRowMajor,
									(transA ? clblast::Transpose::kYes : clblast::Transpose::kNo),
									(transB ? clblast::Transpose::kYes : clblast::Transpose::kNo),
									m,
									n,
									k,
									alpha,
									a(),
									0,
									lda,
									b(),
									0,
									ldb,
									beta,
									c(),
									0,
									ldc,
									&global::openCLQueue());

		LIBRAPID_ASSERT(status == clblast::StatusCode::kSuccess, "clblast::Gemm failed");
	}

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)

	template<typename Backend, typename Int, typename Alpha, typename Beta>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, std::shared_ptr<float> a,
			  Int lda, Beta beta, std::shared_ptr<float> b, Int ldb, std::shared_ptr<float> c,
			  Int ldc, backend::CUDA) {
		cublasSafeCall(cublasSgemm(global::cublasHandle,
								   (transA ? CUBLAS_OP_N : CUBLAS_OP_T),
								   (transB ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   k,
								   &alpha,
								   b.get(),
								   ldb,
								   a.get(),
								   lda,
								   &beta,
								   c.get(),
								   ldc));
	}

	template<typename Backend, typename Int, typename Alpha, typename Beta>
	void gemm(bool transA, bool transB, Int m, Int n, Int k, Alpha alpha, std::shared_ptr<double> a,
			  Int lda, Beta beta, std::shared_ptr<double> b, Int ldb, std::shared_ptr<double> c,
			  Int ldc, backend::CUDA) {
		cublasSafeCall(cublasDgemm(global::cublasHandle,
								   (transA ? CUBLAS_OP_N : CUBLAS_OP_T),
								   (transB ? CUBLAS_OP_N : CUBLAS_OP_T),
								   n,
								   m,
								   k,
								   &alpha,
								   b.get(),
								   ldb,
								   a.get(),
								   lda,
								   &beta,
								   c.get(),
								   ldc));
	}

#endif // LIBRAPID_HAS_CUDA

	template<typename A, typename B,
			 typename Alpha = typename typetraits::TypeInfo<std::decay_t<A>>::Scalar,
			 typename Beta	= typename typetraits::TypeInfo<std::decay_t<B>>::Scalar>
	class Gemm {
	public:
		using ScalarA	= typename typetraits::TypeInfo<std::decay_t<A>>::Scalar;
		using ScalarB	= typename typetraits::TypeInfo<std::decay_t<B>>::Scalar;
		using ShapeType = typename std::decay_t<A>::ShapeType;
		using Backend	= typename typetraits::TypeInfo<std::decay_t<A>>::Backend;
		using BackendB	= typename typetraits::TypeInfo<std::decay_t<B>>::Backend;

		static_assert(std::is_same_v<Backend, BackendB>, "Backend of A and B must match");

		Gemm() = delete;

		Gemm(const Gemm &) = default;

		Gemm(Gemm &&) = default;

		Gemm(bool transA, bool transB, A &a, Alpha alpha, B &b, Beta beta);

		Gemm &operator=(const Gemm &) = default;

		Gemm &operator=(Gemm &&) = default;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ScalarA alpha() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ScalarB beta() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool transA() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool transB() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const A &a() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const B &b() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE A &a();
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE B &b();

		template<typename StorageType>
		void applyTo(ArrayRef<StorageType> &out) const;

	private:
		bool m_transA;
		bool m_transB;
		A &&m_a;
		ScalarA m_alpha;
		B &&m_b;
		ScalarB m_beta;
	};

	template<typename A, typename B, typename Alpha, typename Beta>
	Gemm<A, B, Alpha, Beta>::Gemm(bool transA, bool transB, A &a, Alpha alpha, B &b, Beta beta) :
			m_transA(transA), m_transB(transB), m_a(std::forward<A>(a)),
			m_alpha(static_cast<ScalarA>(alpha)), m_b(std::forward<B>(b)),
			m_beta(static_cast<ScalarB>(beta)) {
		LIBRAPID_ASSERT(m_a.ndim() == 2, "First argument to gemm must be a matrix");
		LIBRAPID_ASSERT(m_b.ndim() == 2, "Second argument to gemm must be a matrix");
		LIBRAPID_ASSERT(m_a.shape()[1 - int(transA)] == m_b.shape()[int(transB)],
						"Inner dimensions of matrices must match");
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::shape() const -> ShapeType {
		return {m_a.shape()[int(m_transA)], m_b.shape()[int(!m_transB)]};
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::alpha() const -> ScalarA {
		return m_alpha;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::beta() const -> ScalarB {
		return m_beta;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::transA() const -> bool {
		return m_transA;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::transB() const -> bool {
		return m_transB;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::a() const -> const A & {
		return m_a;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::b() const -> const B & {
		return m_b;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::a() -> A & {
		return m_a;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	auto Gemm<A, B, Alpha, Beta>::b() -> B & {
		return m_b;
	}

	template<typename A, typename B, typename Alpha, typename Beta>
	template<typename StorageType>
	void Gemm<A, B, Alpha, Beta>::applyTo(ArrayRef<StorageType> &out) const {
		LIBRAPID_ASSERT(out.shape() == shape(), "Output shape must match shape of gemm operation");

		auto m = int64_t(m_a.shape()[m_transA]);
		auto n = int64_t(m_b.shape()[1 - m_transB]);
		auto k = int64_t(m_a.shape()[1 - m_transA]);

		auto lda = int64_t(m_a.shape()[1]);
		auto ldb = int64_t(m_b.shape()[1]);
		auto ldc = int64_t(out.shape()[1]);

		gemm<Backend>(m_transA,
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

#endif // LIBRAPID_ARRAY_LINALG_LEVEL3_GEMM_HPP