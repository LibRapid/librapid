#ifndef LIBRAPID_ARRAY_STRIDE_TOOLS_HPP
#define LIBRAPID_ARRAY_STRIDE_TOOLS_HPP

namespace librapid {
	namespace typetraits {
		LIBRAPID_DEFINE_AS_TYPE(typename ShapeType, Stride<ShapeType>);
	}

	/// A Stride is a vector of integers that describes the distance between elements in each
	/// dimension of an ArrayContainer object. This can be used to access elements in a non-trivial
	/// order, or to access a sub-array of an ArrayContainer object. The Stride class inherits from
	/// the Shape class.
	/// \tparam T The type of the Stride. Must be an integer type.
	/// \tparam N The number of dimensions in the Stride.
	/// \see Shape
	// class Stride : public Shape {
	// public:
	// 	/// Default Constructor
	// 	LIBRAPID_ALWAYS_INLINE Stride() = default;

	// 	/// Construct a Stride from a Shape object. This will assume that the data represented by
	// 	/// the Shape object is a contiguous block of memory, and will calculate the corresponding
	// 	/// strides based on this.
	// 	/// \param shape
	// 	LIBRAPID_ALWAYS_INLINE Stride(const Shape &shape);

	// 	/// Copy a Stride object
	// 	/// \param other The Stride object to copy.
	// 	LIBRAPID_ALWAYS_INLINE Stride(const Stride &other) = default;

	// 	/// Move a Stride object
	// 	/// \param other The Stride object to move.
	// 	LIBRAPID_ALWAYS_INLINE Stride(Stride &&other) noexcept = default;

	// 	/// Assign a Stride object to this Stride object.
	// 	/// \param other The Stride object to assign.
	// 	LIBRAPID_ALWAYS_INLINE Stride &operator=(const Stride &other) = default;

	// 	/// Move a Stride object to this Stride object.
	// 	/// \param other The Stride object to move.
	// 	LIBRAPID_ALWAYS_INLINE Stride &operator=(Stride &&other) noexcept = default;
	// };

	// LIBRAPID_ALWAYS_INLINE Stride::Stride(const Shape &shape) : Shape(shape) {
	// 	if (this->m_dims == 0) {
	// 		// Edge case for a zero-dimensional array
	// 		this->m_data[0] = 1;
	// 		return;
	// 	}

	// 	uint32_t tmp[MaxDimensions] {0};
	// 	tmp[this->m_dims - 1] = 1;
	// 	for (size_t i = this->m_dims - 1; i > 0; --i) tmp[i - 1] = tmp[i] * this->m_data[i];
	// 	for (size_t i = 0; i < this->m_dims; ++i) this->m_data[i] = tmp[i];
	// }

	template<typename ShapeType_>
	class Stride {
	public:
		using ShapeType = ShapeType_;
		using IndexType = typename std::decay_t<decltype(std::declval<ShapeType>()[0])>;
		static constexpr size_t MaxDimensions = ShapeType::MaxDimensions;

		LIBRAPID_ALWAYS_INLINE Stride() = default;
		LIBRAPID_ALWAYS_INLINE Stride(const ShapeType &shape);
		LIBRAPID_ALWAYS_INLINE Stride(const Stride &other)				  = default;
		LIBRAPID_ALWAYS_INLINE Stride(Stride &&other) noexcept			  = default;
		LIBRAPID_ALWAYS_INLINE Stride &operator=(const Stride &other)	  = default;
		LIBRAPID_ALWAYS_INLINE Stride &operator=(Stride &&other) noexcept = default;

		LIBRAPID_ALWAYS_INLINE auto operator[](size_t index) const -> IndexType;
		LIBRAPID_ALWAYS_INLINE auto operator[](size_t index) -> IndexType &;

		LIBRAPID_ALWAYS_INLINE auto ndim() const { return m_data.ndim(); }
		LIBRAPID_ALWAYS_INLINE auto substride(size_t start, size_t end) const -> Stride<Shape>;

		LIBRAPID_ALWAYS_INLINE auto data() const -> const ShapeType &;
		LIBRAPID_ALWAYS_INLINE auto data() -> ShapeType &;

		template<typename T_, typename Char, typename Ctx>
		LIBRAPID_ALWAYS_INLINE void str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const;

	protected:
		ShapeType m_data;
	};

	template<typename ShapeType>
	LIBRAPID_ALWAYS_INLINE Stride<ShapeType>::Stride(const ShapeType &shape) : m_data(shape) {
		if (this->m_data.size() == 0) {
			// Edge case for a zero-dimensional array
			this->m_data[0] = 1;
			return;
		}

		uint32_t tmp[MaxDimensions] {0};
		tmp[shape.ndim() - 1] = 1;
		for (size_t i = shape.ndim() - 1; i > 0; --i) tmp[i - 1] = tmp[i] * this->m_data[i];
		for (size_t i = 0; i < shape.ndim(); ++i) this->m_data[i] = tmp[i];
	}

	template<typename ShapeType>
	LIBRAPID_ALWAYS_INLINE auto Stride<ShapeType>::operator[](size_t index) const -> IndexType {
		return this->m_data[index];
	}

	template<typename ShapeType>
	LIBRAPID_ALWAYS_INLINE auto Stride<ShapeType>::operator[](size_t index) -> IndexType & {
		return this->m_data[index];
	}

	template<typename ShapeType>
	LIBRAPID_ALWAYS_INLINE auto Stride<ShapeType>::substride(size_t start, size_t end) const
	  -> Stride<Shape> {
		LIBRAPID_ASSERT(start < end, "Start index must be less than end index");
		LIBRAPID_ASSERT(end <= this->m_data.ndim(), "End index must be less than ndim()");

		Stride<Shape> res;
		res.data() = data().subshape(start, end);
		return res;
	}

	template<typename ShapeType>
	LIBRAPID_ALWAYS_INLINE auto Stride<ShapeType>::data() const -> const ShapeType & {
		return this->m_data;
	}

	template<typename ShapeType>
	LIBRAPID_ALWAYS_INLINE auto Stride<ShapeType>::data() -> ShapeType & {
		return this->m_data;
	}

	template<typename ShapeType>
	template<typename T_, typename Char, typename Ctx>
	LIBRAPID_ALWAYS_INLINE void Stride<ShapeType>::str(const fmt::formatter<T_, Char> &format,
													   Ctx &ctx) const {
		fmt::format_to(ctx.out(), "Stride(");
		for (size_t i = 0; i < m_data.ndim(); ++i) {
			format.format(m_data[i], ctx);
			if (i != m_data.ndim() - 1) fmt::format_to(ctx.out(), ", ");
		}
		fmt::format_to(ctx.out(), ")");
	}
} // namespace librapid

// Support FMT printing
template<typename T>
struct fmt::formatter<librapid::Stride<T>> : fmt::formatter<librapid::Shape> {
	template<typename FormatContext>
	auto format(const librapid::Stride<T> &stride, FormatContext &ctx) {
		return fmt::formatter<librapid::Shape>::format(stride, ctx);
	}
};

#endif // LIBRAPID_ARRAY_STRIDE_TOOLS_HPP