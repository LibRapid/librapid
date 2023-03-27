#ifndef LIBRAPID_ARRAY_TRANSPOSE_HPP
#define LIBRAPID_ARRAY_TRANSPOSE_HPP

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct TypeInfo<array::Transpose<T>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Transpose;
			using Scalar							   = typename TypeInfo<std::decay_t<T>>::Scalar;
			using Device							   = typename TypeInfo<std::decay_t<T>>::Device;
			static constexpr bool allowVectorisation   = false;
		};
	} // namespace typetraits

	namespace detail {
#define TRANSPOSE_IMPL_KERNEL(KERN_)                                                               \
	for (int64_t i = 0; i < rows; i += blockSize) {                                                \
		for (int64_t j = 0; j < cols; j += blockSize) {                                            \
			for (int64_t row = i; row < i + blockSize && row < rows; ++row) {                      \
				for (int64_t col = j; col < j + blockSize && col < cols; ++col) { KERN_; }         \
		}}                                                                                         \
	}                                                                                              \
	do {                                                                                           \
	} while (false)

		template<typename Scalar>
		LIBRAPID_ALWAYS_INLINE void transposeImpl(Scalar *__restrict out, Scalar *__restrict in,
												  int64_t rows, int64_t cols, int64_t blockSize) {
			if (rows * cols < global::multithreadThreshold) {
				TRANSPOSE_IMPL_KERNEL(out[col * rows + row] = in[row * cols + col]);
			} else {
#pragma omp parallel for shared(rows, cols, blockSize, in, out) default(none)                      \
  num_threads((int)global::numThreads)
				TRANSPOSE_IMPL_KERNEL(out[col * rows + row] = in[row * cols + col]);
			}
		}

		template<typename Scalar>
		LIBRAPID_ALWAYS_INLINE void transposeInplaceImpl(Scalar *__restrict data, int64_t rows,
														 int64_t cols, int64_t blockSize) {
			if (rows * cols < global::multithreadThreshold) {
				TRANSPOSE_IMPL_KERNEL(std::swap(data[col * rows + row], data[row * cols + col]););
			} else {
#pragma omp parallel for shared(rows, cols, blockSize, data) default(none)                      \
  num_threads((int)global::numThreads)
				TRANSPOSE_IMPL_KERNEL(std::swap(data[col * rows + row], data[row * cols + col]););
			}
		}

#undef TRANSPOSE_IMPL_KERNEL
	} // namespace detail

	namespace array {
		template<typename T>
		class Transpose {
		public:
			using ArrayType		 = T;
			using BaseType		 = typename std::decay_t<T>;
			using Scalar		 = typename typetraits::TypeInfo<BaseType>::Scalar;
			using Reference		 = BaseType &;
			using ConstReference = const BaseType &;
			using ShapeType		 = typename BaseType::ShapeType;
			using Device		 = typename typetraits::TypeInfo<BaseType>::Device;

			static constexpr bool allowVectorisation =
			  typetraits::TypeInfo<Scalar>::allowVectorisation;
			static constexpr bool isArray = typetraits::IsArrayContainer<BaseType>::value;
			static constexpr bool isHost  = std::is_same_v<Device, device::CPU>;

			/// Default constructor should never be used
			Transpose() = delete;

			/// Create a Transpose object from an array/operation
			/// \param array The array to copy
			/// \param axes The transposition axes
			explicit Transpose(T &array, const ShapeType &axes);

			/// Copy a Transpose object
			Transpose(const Transpose &other) = default;

			/// Move constructor
			Transpose(Transpose &&other) = default;

			/// Assign another Transpose object to this one
			/// \param other The Transpose to assign
			/// \return *this;
			Transpose &operator=(const Transpose &other) = default;

			/// Access sub-array of this Transpose object
			/// \param index Array index
			/// \return ArrayView<T>
			ArrayView<ArrayType> operator[](int64_t index) const;

			/// Get the shape of this Transpose object
			/// \return Shape
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

			/// Return the number of dimensions of the Transpose object
			/// \return Number of dimensions
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t ndim() const;

			/// Access a scalar at a given index in the object. The index will be converted into
			/// a multi-dimensional index using the shape of the object, and counts in row-major
			/// order
			/// \param index Index of the scalar
			/// \return Scalar type at the given index
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalar(int64_t index) const;

			template<typename Container>
			LIBRAPID_ALWAYS_INLINE void applyTo(Container &out) const;

			/// Evaluate the Transpose object and return the result. Depending on your use case,
			/// calling this function mid-expression might result in better performance, but you
			/// should always test the available options before making a decision.
			/// \return Evaluated expression
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto eval() const;

			/// Return a string representation of the Transpose object, formatting each scalar with
			/// the given format string
			/// \param format Format string
			/// \return Stringified object
			LIBRAPID_NODISCARD std::string str(const std::string &format = "{}") const;

		private:
			ArrayType &m_array;
			ShapeType m_inputShape;
			ShapeType m_outputShape;
			ShapeType m_axes;
		};

		template<typename T>
		Transpose<T>::Transpose(T &array, const ShapeType &axes) :
				m_array(array), m_inputShape(array.shape()), m_axes(axes) {
			LIBRAPID_ASSERT(m_inputShape.ndim() == m_axes.ndim(),
							"Shape and axes must have the same number of dimensions");

			m_outputShape = m_inputShape;
			for (int64_t i = 0; i < m_inputShape.ndim(); i++) {
				m_outputShape[i] = m_inputShape[m_axes[i]];
			}
		}

		template<typename T>
		auto Transpose<T>::shape() const -> ShapeType {
			return m_outputShape;
		}

		template<typename T>
		template<typename Container>
		void Transpose<T>::applyTo(Container &out) const {
			LIBRAPID_ASSERT(out.shape() == m_outputShape, "Transpose assignment shape mismatch");
			bool inplace = &out == &m_array;

			if constexpr (isArray && isHost && allowVectorisation) {
				auto *__restrict outPtr = out.storage().begin();
				auto *__restrict inPtr	= m_array.storage().begin();
				int64_t blockSize		= global::cacheLineSize / sizeof(Scalar);

				if (m_inputShape.ndim() == 2) {
					if (inplace) {
						detail::transposeInplaceImpl(
						  outPtr, m_inputShape[0], m_inputShape[1], blockSize);
					} else {
						detail::transposeImpl(
						  outPtr, inPtr, m_inputShape[0], m_inputShape[1], blockSize);
					}
				} else {
					LIBRAPID_NOT_IMPLEMENTED
				}
			} else {
				LIBRAPID_NOT_IMPLEMENTED
			}
		}

		template<typename T>
		auto Transpose<T>::eval() const {
			ArrayType res(m_outputShape);
			applyTo(res);
			return res;
		}
	}; // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_TRANSPOSE_HPP
