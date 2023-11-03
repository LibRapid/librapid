#ifndef LIBRAPID_ARRAY_LINALG_ARRAY_MULTIPLY_HPP
#define LIBRAPID_ARRAY_LINALG_ARRAY_MULTIPLY_HPP

namespace librapid {
	namespace detail {
		/// Extract the pointer from a given array type
		/// \tparam T
		/// \param ptr
		/// \return
		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto arrayPointerExtractor(T *ptr) {
			return ptr;
		}

#if defined(LIBRAPID_HAS_OPENCL)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto arrayPointerExtractor(cl::Buffer ptr) {
			return ptr;
		}
#endif // LIBRAPID_HAS_OPENCL

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		arrayPointerExtractor(std::shared_ptr<T> ptr) {
			return ptr.get();
		}
	} // namespace detail

	namespace linalg {
		enum class MatmulClass {
			DOT,   // Vector-vector dot product
			GEMV,  // Matrix-vector product
			GEMM,  // Matrix-matrix product
			OUTER, // Outer product
		};

		/// Class to represent an array multiplication (vector-vector, matrix-vector, matrix-matrix)
		/// \tparam ShapeTypeA Shape of the first array
		/// \tparam StorageTypeA Storage type of the first array
		/// \tparam ShapeTypeB Shape of the second array
		/// \tparam StorageTypeB Storage type of the second array
		/// \tparam Alpha Type of \f$ \alpha \f$ scaling factor
		/// \tparam Beta Type of \f$ \beta \f$ scaling factor
		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		class ArrayMultiply {
		public:
			using TypeA		= array::ArrayContainer<ShapeTypeA, StorageTypeA>;
			using TypeB		= array::ArrayContainer<ShapeTypeB, StorageTypeB>;
			using ScalarA	= typename StorageTypeA::Scalar;
			using ScalarB	= typename StorageTypeB::Scalar;
			using Scalar	= decltype(std::declval<ScalarA>() * std::declval<ScalarB>());
			using ShapeType = ShapeTypeA;
			using BackendA	= typename typetraits::TypeInfo<TypeA>::Backend;
			using BackendB	= typename typetraits::TypeInfo<TypeB>::Backend;
			using Backend	= decltype(typetraits::commonBackend<BackendA, BackendB>());

			static_assert(std::is_same_v<Backend, BackendB>, "Backend of A and B must match");

			/// Default constructor (deleted)
			ArrayMultiply() = delete;

			/// Copy constructor
			ArrayMultiply(const ArrayMultiply &) = default;

			/// Move constructor
			ArrayMultiply(ArrayMultiply &&) noexcept = default;

			/// \brief Full set of parameters
			/// \param transA
			/// \param transB
			/// \param a
			/// \param alpha
			/// \param b
			/// \param beta
			ArrayMultiply(bool transA, bool transB, TypeA &&a, Alpha alpha, TypeB &&b, Beta beta);

			/// \brief Array multiplication with \f$ \alpha = 1 \f$ and \f$ \beta = 0 \f$
			/// \param a
			/// \param b
			ArrayMultiply(TypeA &&a, TypeB &&b);

			/// \brief Array multiplication with \f$ \alpha = 1 \f$ and \f$ \beta = 0 \f$ and
			/// transpose options
			/// \param transA
			/// \param transB
			/// \param a
			/// \param b
			ArrayMultiply(bool transA, bool transB, TypeA &&a, TypeB &&b);

			/// \brief Copy assignment operator
			/// \return Reference to this
			ArrayMultiply &operator=(const ArrayMultiply &) = default;

			/// \brief Move assignment operator
			/// \return Reference to this
			ArrayMultiply &operator=(ArrayMultiply &&) noexcept = default;

			/// \brief Determine the class of the array multiplication
			///
			/// The class of the array multiplication is determined by the shapes of the arrays.
			/// There are three supported cases:
			/// - Vector-vector dot product (both arrays are 1-dimensional vectors)
			/// - Matrix-vector product (first array is a 2-dimensional matrix, second array is a
			/// 1-dimensional vector)
			/// - Matrix-matrix product (both arrays are 2-dimensional matrices)
			/// \return Class of the array multiplication
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE MatmulClass matmulClass() const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType calculateShape() const;

			/// \brief Determine the shape of the result
			/// \return Shape of the result
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE size_t size() const;

			/// \brief Determine the number of dimensions of the result
			/// \return Number of dimensions of the result
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t ndim() const;

			/// \brief Force evaluation of the array multiplication, returning an Array object
			/// \return Array object containing the result
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto eval() const;

			/// \brief Get the scaling factor \f$ \alpha \f$
			/// \return \f$ \alpha \f$
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ScalarA alpha() const;

			/// \brief Get the scaling factor \f$ \beta \f$
			/// \return \f$ \beta \f$
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ScalarB beta() const;

			/// \brief Determine \f$ \mathrm{OP}_A \f$
			/// \return True: \f$ \mathrm{OP}_A(\mathbf{A}) = \mathbf{A}^T \f$, false: \f$
			/// \mathrm{OP}_A(\mathbf{A}) = \mathbf{A} \f$
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool transA() const;

			/// \brief Determine \f$ \mathrm{OP}_B \f$
			/// \return True: \f$ \mathrm{OP}_B(\mathbf{B}) = \mathbf{B}^T \f$, false: \f$
			/// \mathrm{OP}_B(\mathbf{B}) = \mathbf{B} \f$
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool transB() const;

			/// \brief Get the first array
			/// \return First array
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const TypeA &a() const;

			/// \brief Get the second array
			/// \return Second array
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const TypeB &b() const;

			/// \brief Get the first array
			/// \return First array
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE TypeA &a();

			/// \brief Get the second array
			/// \return Second array
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE TypeB &b();

			/// \brief Apply the array multiplication to an array container
			///
			/// Apply this operation to the provided Array, assuming that the Array has the correct
			/// shape. If the Array does not have the correct shape, an error is thrown.
			///
			/// \tparam StorageType Storage type of the array container
			/// \param out Array container to store the result in
			template<typename StorageType>
			void applyTo(array::ArrayContainer<ShapeType, StorageType> &out) const;

			template<typename T, typename Char, size_t N, typename Ctx>
			void str(const fmt::formatter<T, Char> &format, char bracket, char separator,
					 const char (&formatString)[N], Ctx &ctx) const;

		private:
			bool m_transA;	 // Transpose state of A
			bool m_transB;	 // Transpose state of B
			TypeA m_a;		 // First array
			ScalarA m_alpha; // Scaling factor for A
			TypeB m_b;		 // Second array
			ScalarB m_beta;	 // Scaling factor for B

			ShapeType m_shape;
			size_t m_size;
		};

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
					  Beta>::ArrayMultiply(bool transA, bool transB, TypeA &&a, Alpha alpha,
										   TypeB &&b, Beta beta) :
				m_transA(transA),
				m_transB(transB), m_a(std::forward<TypeA>(a)), m_alpha(static_cast<ScalarA>(alpha)),
				m_b(std::forward<TypeB>(b)), m_beta(static_cast<ScalarB>(beta)),
				m_shape(calculateShape()), m_size(m_shape.size()) {}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
					  Beta>::ArrayMultiply(TypeA &&a, TypeB &&b) :
				m_transA(false),
				m_transB(false), m_a(std::forward<TypeA>(a)), m_alpha(1),
				m_b(std::forward<TypeB>(b)), m_beta(0), m_shape(calculateShape()),
				m_size(m_shape.size()) {}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
					  Beta>::ArrayMultiply(bool transA, bool transB, TypeA &&a, TypeB &&b) :
				m_transA(transA),
				m_transB(transB), m_a(std::forward<TypeA>(a)), m_alpha(1),
				m_b(std::forward<TypeB>(b)), m_beta(0), m_shape(calculateShape()),
				m_size(m_shape.size()) {}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
						   Beta>::matmulClass() const -> MatmulClass {
			const auto &shapeA = m_a.shape();
			const auto &shapeB = m_b.shape();

			if (shapeA.ndim() == 1 && shapeB.ndim() == 1) {
				LIBRAPID_ASSERT(shapeA[0] == shapeB[0],
								"Vector dimensions must. Expected: {} -- Got: {}",
								shapeA[0],
								shapeB[0]);

				return MatmulClass::DOT;
			} else if (shapeA.ndim() == 1 && shapeB.ndim() == 2) {
				LIBRAPID_ASSERT(
				  shapeA[0] == shapeB[int(!m_transB)],
				  "Columns of OP(B) must match elements of A. Expected: {} -- Got: {}",
				  shapeA[0],
				  shapeB[int(!m_transB)]);

				return MatmulClass::GEMV;
			} else if (shapeA.ndim() == 2 && shapeB.ndim() == 1) {
				LIBRAPID_ASSERT(shapeA[int(!m_transA)] == shapeB[0],
								"Rows of OP(A) must match elements of B. Expected: {} -- Got: {}",
								shapeA[int(m_transA)],
								shapeB[0]);

				return MatmulClass::GEMV;
			} else if (shapeA.ndim() == 2 && shapeB.ndim() == 2) {
				// // Check for GEMV variations
				// // 1. A is a matrix, B is a 1xn vector
				// // 2. A is a matrix, B is a nx1 vector

				// if (shapeB[0] == 1) { // Case 1
				// 	LIBRAPID_ASSERT(
				// 	  shapeA[int(!m_transA)] == shapeB[1],
				// 	  "Columns of {} must match columns of B. Expected: {} -- Got: {}",
				// 	  (m_transA ? "A" : "A^T"),
				// 	  shapeA[int(!m_transA)],
				// 	  shapeB[1]);

				// 	return MatmulClass::GEMV;
				// } else if (shapeB[1] == 1) { // Case 2
				// 	LIBRAPID_ASSERT(shapeA[int(!m_transA)] == shapeB[0],
				// 					"Columns of {} must match rows of B. Expected: {} -- Got: {}",
				// 					(m_transA ? "A" : "A^T"),
				// 					shapeA[int(!m_transA)],
				// 					shapeB[0]);

				// 	return MatmulClass::GEMV;
				// }

				LIBRAPID_ASSERT(m_a.shape()[int(!m_transA)] == m_b.shape()[int(m_transB)],
								"Inner dimensions of matrices must match. Expected: {} -- Got: {}",
								m_a.shape()[int(!m_transA)],
								m_b.shape()[int(m_transB)]);

				return MatmulClass::GEMM;
			} else {
				LIBRAPID_NOT_IMPLEMENTED;

				return MatmulClass::OUTER;
			}
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
						   Beta>::calculateShape() const -> ShapeType {
			const auto &shapeA		= m_a.shape();
			const auto &shapeB		= m_b.shape();
			MatmulClass matmulClass = this->matmulClass();

			switch (matmulClass) {
				case MatmulClass::DOT: {
					return {1};
				}
				case MatmulClass::GEMV: {
					return {shapeA[int(m_transA)]};
				}
				case MatmulClass::GEMM: {
					return {shapeA[int(m_transA)], shapeB[int(!m_transB)]};
				}
				case MatmulClass::OUTER: {
					LIBRAPID_NOT_IMPLEMENTED;
					return {1};
				}
			}

			LIBRAPID_NOT_IMPLEMENTED;
			return {1};
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::shape()
		  const -> ShapeType {
			return m_shape;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::size() const
		  -> size_t {
			return m_size;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::ndim() const
		  -> int64_t {
			return shape().ndim();
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::eval()
		  const {
			Array<Scalar, Backend> result(shape());
			applyTo(result);
			return result;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::alpha()
		  const -> ScalarA {
			return m_alpha;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::beta() const
		  -> ScalarB {
			return m_beta;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		bool
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::transA()
		  const {
			return m_transA;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		bool
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::transB()
		  const {
			return m_transB;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::a() const
		  -> const TypeA & {
			return m_a;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::b() const
		  -> const TypeB & {
			return m_b;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::a()
		  -> TypeA & {
			return m_a;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		auto ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::b()
		  -> TypeB & {
			return m_b;
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		template<typename StorageType>
		void
		ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::applyTo(
		  array::ArrayContainer<ShapeType, StorageType> &out) const {
			LIBRAPID_ASSERT(out.shape() == shape(),
							"Shape of output array must match shape of array multiply operation. "
							"Expected: {} -- Got: {}",
							shape(),
							out.shape());
			MatmulClass matmulClass = this->matmulClass();

			auto a = detail::arrayPointerExtractor(m_a.storage().data());
			auto b = detail::arrayPointerExtractor(m_b.storage().data());
			auto c = detail::arrayPointerExtractor(out.storage().data());

			switch (matmulClass) {
				case MatmulClass::DOT: {
					LIBRAPID_NOT_IMPLEMENTED;
				}
				case MatmulClass::GEMV: {
					auto m = int64_t(m_a.shape()[m_transA]);
					auto n = int64_t(m_a.shape()[1 - m_transA]);

					auto lda  = int64_t(m_a.shape()[1]);
					auto incB = int64_t(1);
					auto incC = int64_t(1);

					gemv(m_transA,
						 m,
						 n,
						 static_cast<Scalar>(m_alpha),
						 a,
						 lda,
						 b,
						 incB,
						 static_cast<Scalar>(m_beta),
						 c,
						 incC,
						 Backend());

					break;
				}
				case MatmulClass::GEMM: {
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
						 static_cast<Scalar>(m_alpha),
						 a,
						 lda,
						 b,
						 ldb,
						 static_cast<Scalar>(m_beta),
						 c,
						 ldc,
						 Backend());

					break;
				}
				default: {
					LIBRAPID_NOT_IMPLEMENTED;
				}
			}
		}

		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		template<typename T, typename Char, size_t N, typename Ctx>
		void ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>::str(
		  const fmt::formatter<T, Char> &format, char bracket, char separator,
		  const char (&formatString)[N], Ctx &ctx) const {
			eval().str(format, bracket, separator, formatString, ctx);
		}
	} // namespace linalg

	//	/// \brief Computes the dot product of two arrays.
	//	///
	//	/// This function calculates the dot product of two arrays.
	//	///
	//	/// If the input arrays are 1-dimensional vectors, this function computes the vector-dot
	//	/// product \f$ \mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + \ldots + a_nb_n \f$.
	//	/// Note that the return value will be a 1x1 array (i.e. a scalar) since we cannot return a
	//	/// scalar directly.
	//	///
	//	/// If the left input is a 2-dimensional matrix and the right input is a 1-dimensional
	// vector,
	//	/// this function computes the matrix-vector product \f$ y_i = \sum_{j=1}^{n} a_{ij} x_j \f$
	//	/// for \f$ i = 1, \ldots, m \f$.
	//	///
	//	/// If both inputs are 2-dimensional matrices, this function computes the matrix-matrix
	// product
	//	/// \f$ c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} \f$ for \f$ i = 1, \ldots, m \f$ and \f$ j =
	// 1,
	//	/// \ldots, p \f$. \tparam StorageTypeA The storage type of the left input array. \tparam
	//	/// StorageTypeB The storage type of the right input array. \param a The left input array.
	//	/// \param b The right input array.
	//	/// \return The dot product of the two input arrays.
	//	template<typename StorageTypeA, typename StorageTypeB>
	//	auto dot(const ArrayRef<StorageTypeA> &a, const ArrayRef<StorageTypeB> &b) {
	//		return linalg::ArrayMultiply(a, b);
	//	}

	namespace detail {
		template<typename ShapeType, typename DestinationStorageType, typename ShapeTypeA,
				 typename StorageTypeA, typename ShapeTypeB, typename StorageTypeB,
				 typename Alpha = typename StorageTypeA::Scalar,
				 typename Beta	= typename StorageTypeB::Scalar>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType, DestinationStorageType> &destination,
			   const linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB,
										   Alpha, Beta> &op) {
			op.applyTo(destination);
		}

		/// Evaluates to true if the type is a transpose type.
		/// \tparam T
		template<typename T>
		struct IsTransposeType : std::false_type {};

		template<typename T>
		struct IsTransposeType<array::Transpose<T>> : std::true_type {};

		/// Returns a tuple of the form (transpose, raw array) where transpose is true if the array
		/// is transposed and false otherwise, and raw array is the raw array data.
		/// \tparam T
		/// \param val
		/// \return
		template<typename T, typename std::enable_if_t<!IsTransposeType<T>::value, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto transposeExtractor(T &&val) {
			using Scalar = typename typetraits::TypeInfo<std::decay_t<T>>::Scalar;
			return std::make_tuple(false, Scalar(1), std::forward<T>(val));
		}

		template<typename T, typename std::enable_if_t<IsTransposeType<T>::value, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto transposeExtractor(T &&val) {
			using Type = decltype(val.array());
			return std::make_tuple(true, val.alpha(), std::forward<Type>(val.array()));
		}

		/// Evaluates to true if the type is a multiply type.
		/// \tparam T
		template<typename T>
		struct IsMultiplyType : std::false_type {};

		template<typename Descriptor, typename Arr, typename Scalar>
		struct IsMultiplyType<detail::Function<Descriptor, detail::Multiply, Arr, Scalar>>
				: std::true_type {};

		/// Returns a tuple of the form (scalar, raw array) where scalar is the multiplication
		/// factor and raw array is the raw array data.
		/// \tparam T
		/// \param val
		/// \return
		template<typename T, typename std::enable_if_t<!IsMultiplyType<T>::value, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto multiplyExtractor(T &&val) {
			using Scalar = typename typetraits::TypeInfo<std::decay_t<T>>::Scalar;
			return std::make_tuple(Scalar(1), std::forward<T>(val));
		}

		template<typename Descriptor, typename Arr, typename Scalar,
				 typename std::enable_if_t<
				   typetraits::TypeInfo<Scalar>::type == detail::LibRapidType::Scalar, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		multiplyExtractor(detail::Function<Descriptor, detail::Multiply, Arr, Scalar> &&val) {
			using Type = decltype(std::get<0>(val.args()));
			return std::make_tuple(std::get<1>(val.args()),
								   std::forward<Type>(std::get<0>(val.args())));
		}

		template<typename Descriptor, typename Arr, typename Scalar,
				 typename std::enable_if_t<
				   typetraits::TypeInfo<Scalar>::type == detail::LibRapidType::Scalar, int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		multiplyExtractor(detail::Function<Descriptor, detail::Multiply, Scalar, Arr> &&val) {
			using Type = decltype(std::get<1>(val.args()));
			return std::make_tuple(std::get<0>(val.args()),
								   std::forward<Type>(std::get<1>(val.args())));
		}

		/// Return a tuple of the form (transpose, scalar, raw array) depending on the input type.
		/// All scalar values are extracted and combined, and successive transpose operations are
		/// combined.
		/// \tparam T
		/// \param val
		/// \return
		template<typename T>
		auto dotHelper(T &&val) {
			if constexpr (IsTransposeType<T>::value) {
				auto [transpose, alpha, array]	  = transposeExtractor(std::forward<T>(val));
				using ArrayType					  = decltype(array);
				auto [transpose2, alpha2, array2] = dotHelper(std::forward<ArrayType>(array));
				using Array2Type				  = decltype(array2);
				return std::make_tuple(
				  transpose ^ transpose2, alpha * alpha2, std::forward<Array2Type>(array2));
			} else if constexpr (IsMultiplyType<T>::value) {
				auto [alpha, array]				 = multiplyExtractor(std::forward<T>(val));
				using ArrayType					 = decltype(array);
				auto [transpose, alpha2, array2] = dotHelper(std::forward<ArrayType>(array));
				using Array2Type				 = decltype(array2);
				return std::make_tuple(transpose, alpha * alpha2, std::forward<Array2Type>(array2));
			} else {
				using Scalar = typename typetraits::TypeInfo<std::decay_t<T>>::Scalar;
				return std::make_tuple(false, Scalar(1), std::forward<T>(val));
			}
		}
	} // namespace detail

	/// \brief Computes the dot product of two arrays.
	///
	/// This function calculates the dot product of two arrays.
	///
	/// If the input arrays are 1-dimensional vectors, this function computes the vector-dot
	/// product \f$ \mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + \ldots + a_nb_n \f$.
	/// Note that the return value will be a 1x1 array (i.e. a scalar) since we cannot return a
	/// scalar directly.
	///
	/// If the left input is a 2-dimensional matrix and the right input is a 1-dimensional vector,
	/// this function computes the matrix-vector product \f$ y_i = \sum_{j=1}^{n} a_{ij} x_j \f$
	/// for \f$ i = 1, \ldots, m \f$.
	///
	/// If both inputs are 2-dimensional matrices, this function computes the matrix-matrix product
	/// \f$ c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} \f$ for \f$ i = 1, \ldots, m \f$ and \f$ j = 1,
	/// \ldots, p \f$. \tparam StorageTypeA The storage type of the left input array. \tparam
	/// StorageTypeB The storage type of the right input array. \param a The left input array.
	/// \param b The right input array.
	/// \return The dot product of the two input arrays.
	template<
	  typename First, typename Second,
	  typename std::enable_if_t<IsArrayType<First>::value && IsArrayType<Second>::value, int> = 0>
	auto dot(First &&a, Second &&b) {
		using ScalarA	   = typename typetraits::TypeInfo<std::decay_t<First>>::Scalar;
		using ScalarB	   = typename typetraits::TypeInfo<std::decay_t<Second>>::Scalar;
		using BackendA	   = typename typetraits::TypeInfo<std::decay_t<First>>::Backend;
		using BackendB	   = typename typetraits::TypeInfo<std::decay_t<Second>>::Backend;
		using ArrayA	   = Array<ScalarA, BackendA>;
		using ArrayB	   = Array<ScalarB, BackendB>;
		using ShapeTypeA   = typename typetraits::TypeInfo<std::decay_t<First>>::ShapeType;
		using ShapeTypeB   = typename typetraits::TypeInfo<std::decay_t<Second>>::ShapeType;
		using StorageTypeA = typename typetraits::TypeInfo<std::decay_t<First>>::StorageType;
		using StorageTypeB = typename typetraits::TypeInfo<std::decay_t<Second>>::StorageType;

		auto [transA, alpha, arrA] = detail::dotHelper(std::forward<First>(a));
		auto [transB, beta, arrB]  = detail::dotHelper(std::forward<Second>(b));
		return linalg::
		  ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, ScalarA, ScalarB>(
			transA,
			transB,
			std::forward<ArrayA>(arrA),
			alpha * beta,
			std::forward<ArrayB>(arrB),
			ScalarA(0));
	}

	namespace typetraits {
		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		struct TypeInfo<
		  linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha, Beta>> {
			detail::LibRapidType type = detail::LibRapidType::ArrayFunction;
			using Type	 = linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB,
												 Alpha, Beta>;
			using Scalar = typename Type::Scalar;
			using Backend							 = typename Type::Backend;
			static constexpr bool allowVectorisation = false;
		};

		LIBRAPID_DEFINE_AS_TYPE(typename ShapeTypeA COMMA typename StorageTypeA COMMA
								typename ShapeTypeB COMMA typename StorageTypeB COMMA
								typename Alpha COMMA typename Beta,
								linalg::ArrayMultiply<ShapeTypeA COMMA StorageTypeA COMMA ShapeTypeB
														COMMA StorageTypeB COMMA Alpha COMMA Beta>);
	} // namespace typetraits
} // namespace librapid

ARRAY_TYPE_FMT_IML(typename ShapeTypeA COMMA typename StorageTypeA COMMA typename ShapeTypeB COMMA
				   typename StorageTypeB COMMA typename Alpha COMMA typename Beta,
				   librapid::linalg::ArrayMultiply<ShapeTypeA COMMA StorageTypeA COMMA ShapeTypeB
													 COMMA StorageTypeB COMMA Alpha COMMA Beta>)

LIBRAPID_SIMPLE_IO_NORANGE(
  typename ShapeTypeA COMMA typename StorageTypeA COMMA typename ShapeTypeB COMMA
  typename StorageTypeB COMMA typename Alpha COMMA typename Beta,
  librapid::linalg::ArrayMultiply<
	ShapeTypeA COMMA StorageTypeA COMMA ShapeTypeB COMMA StorageTypeB COMMA Alpha COMMA Beta>)

#endif // LIBRAPID_ARRAY_LINALG_ARRAY_MULTIPLY_HPP