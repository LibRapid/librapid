#ifndef LIBRAPID_ARRAY_STRIDE_TOOLS_HPP
#define LIBRAPID_ARRAY_STRIDE_TOOLS_HPP

namespace librapid {
	namespace typetraits {
		LIBRAPID_DEFINE_AS_TYPE_NO_TEMPLATE(Stride);
	}

	/// A Stride is a vector of integers that describes the distance between elements in each
	/// dimension of an ArrayContainer object. This can be used to access elements in a non-trivial
	/// order, or to access a sub-array of an ArrayContainer object. The Stride class inherits from
	/// the Shape class.
	/// \tparam T The type of the Stride. Must be an integer type.
	/// \tparam N The number of dimensions in the Stride.
	/// \see Shape
	class Stride : public Shape {
	public:
		/// Default Constructor
		LIBRAPID_ALWAYS_INLINE Stride() = default;

		/// Construct a Stride from a Shape object. This will assume that the data represented by
		/// the Shape object is a contiguous block of memory, and will calculate the corresponding
		/// strides based on this.
		/// \param shape
		LIBRAPID_ALWAYS_INLINE Stride(const Shape &shape);

		/// Copy a Stride object
		/// \param other The Stride object to copy.
		LIBRAPID_ALWAYS_INLINE Stride(const Stride &other) = default;

		/// Move a Stride object
		/// \param other The Stride object to move.
		LIBRAPID_ALWAYS_INLINE Stride(Stride &&other) noexcept = default;

		/// Assign a Stride object to this Stride object.
		/// \param other The Stride object to assign.
		LIBRAPID_ALWAYS_INLINE Stride &operator=(const Stride &other) = default;

		/// Move a Stride object to this Stride object.
		/// \param other The Stride object to move.
		LIBRAPID_ALWAYS_INLINE Stride &operator=(Stride &&other) noexcept = default;
	};

	LIBRAPID_ALWAYS_INLINE Stride::Stride(const Shape &shape) : Shape(shape) {
		if (this->m_dims == 0) {
			// Edge case for a zero-dimensional array
			this->m_data[0] = 1;
			return;
		}

		typename Shape::SizeType tmp[MaxDimensions] {0};
		tmp[this->m_dims - 1] = 1;
		for (size_t i = this->m_dims - 1; i > 0; --i) tmp[i - 1] = tmp[i] * this->m_data[i];
		for (size_t i = 0; i < this->m_dims; ++i) this->m_data[i] = tmp[i];
	}
} // namespace librapid

// Support FMT printing
template<>
struct fmt::formatter<librapid::Stride> : fmt::formatter<librapid::Shape> {
	template<typename FormatContext>
	auto format(const librapid::Stride &stride, FormatContext &ctx) {
		return fmt::formatter<librapid::Shape>::format(stride, ctx);
	}
};

#endif // LIBRAPID_ARRAY_STRIDE_TOOLS_HPP