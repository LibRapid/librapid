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
			ShapeType m_shape;
			ShapeType m_axes;
		};

		template<typename T>
		Transpose<T>::Transpose(T &array, const ShapeType &axes) :
				m_array(array), m_shape(array.shape()), m_axes(axes) {
			LIBRAPID_ASSERT(m_shape.size() == m_axes.size(),
							"Shape and axes must have the same number of dimensions");

			for (int64_t i = 0; i < m_shape.size(); i++) { m_shape[i] = array.shape()[m_axes[i]];}


		}
	}; // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_TRANSPOSE_HPP
