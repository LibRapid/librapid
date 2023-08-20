#ifndef LIBRAPID_ARRAY_SIZETYPE_HPP
#define LIBRAPID_ARRAY_SIZETYPE_HPP

/*
 * This file defines the Shape class and some helper functions,
 * including stride operations.
 */

namespace librapid {
	namespace typetraits {
		LIBRAPID_DEFINE_AS_TYPE(size_t N, Shape<N>);
		LIBRAPID_DEFINE_AS_TYPE_NO_TEMPLATE(MatrixShape);
	}

	template<size_t N = LIBRAPID_MAX_ARRAY_DIMS>
	class Shape {
	public:
		using SizeType						  = uint32_t;
		static constexpr size_t MaxDimensions = N;

		/// Default constructor
		Shape() = default;

		/// Create a shape object from the dimensions of a FixedStorage object. This is used
		// mainly internally, but may serve some purpose I haven't yet thought of.
		/// \tparam Scalar Scalar type of the FixedStorage object
		/// \tparam Dimensions Dimensions of the FixedStorage object
		/// \param fixed The FixedStorage object
		template<typename Scalar, size_t... Dimensions>
		explicit Shape(const FixedStorage<Scalar, Dimensions...> &fixed);

		/// Create a Shape object from a list of values
		/// \tparam V Scalar type of the values
		/// \param vals The dimensions for the object
		template<typename V>
		Shape(const std::initializer_list<V> &vals);

		/// Create a Shape object from a vector of values
		/// \tparam V Scalar type of the values
		/// \param vals The dimensions for the object
		template<typename V>
		explicit Shape(const std::vector<V> &vals);

		/// Create a copy of a Shape object
		/// \param other Shape object to copy
		Shape(const Shape &other) = default;

		/// Create a Shape from an RValue
		/// \param other Temporary Shape object to copy
		Shape(Shape &&other) noexcept = default;

		/// Create a Shape object from one with a different type and number of dimensions.
		/// \tparam V Scalar type of the values
		/// \tparam Dim	Number of dimensions
		/// \param other Shape object to copy
		template<size_t Dim>
		Shape(const Shape<Dim> &other);

		/// Create a Shape object from one with a different type and number of dimensions, moving it
		/// instead of copying it.
		/// \tparam V Scalar type of the values
		/// \tparam Dim Number of dimensions
		/// \param other Temporary Shape object to move
		template<size_t Dim>
		Shape(Shape<Dim> &&other) noexcept;

		/// Assign a Shape object to this object
		/// \tparam V Scalar type of the Shape
		/// \param vals Dimensions of the Shape
		/// \return *this
		template<typename V>
		auto operator=(const std::initializer_list<V> &vals) -> Shape &;

		/// Assign a Shape object to this object
		/// \tparam V Scalar type of the Shape
		/// \param vals Dimensions of the Shape
		/// \return *this
		template<typename V>
		auto operator=(const std::vector<V> &vals) -> Shape &;

		/// Assign an RValue Shape to this object
		/// \param other RValue to move
		/// \return
		auto operator=(Shape &&other) noexcept -> Shape & = default;

		/// Assign a Shape to this object
		/// \param other Shape to copy
		/// \return
		auto operator=(const Shape &other) -> Shape & = default;

		/// Return a Shape object with \p dims dimensions, all initialized to zero.
		/// \param dims Number of dimensions
		/// \return New Shape object
		LIBRAPID_ALWAYS_INLINE static auto zeros(size_t dims) -> Shape;

		/// Return a Shape object with \p dims dimensions, all initialized to one.
		/// \param dims Number of dimensions
		/// \return New Shape object
		LIBRAPID_ALWAYS_INLINE static auto ones(size_t dims) -> Shape;

		/// Access an element of the Shape object
		/// \tparam Index Typename of the index
		/// \param index Index to access
		/// \return The value at the index
		template<typename Index>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](Index index) const
		  -> const SizeType &;

		/// Access an element of the Shape object
		/// \tparam Index Typename of the index
		/// \param index Index to access
		/// \return A reference to the value at the index
		template<typename Index>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](Index index) -> SizeType &;

		/// Compare two Shape objects, returning true if and only if they are identical
		/// \param other Shape object to compare
		/// \return	true if the objects are identical
		LIBRAPID_ALWAYS_INLINE auto operator==(const Shape &other) const -> bool;

		/// Compare two Shape objects, returning true if and only if they are not identical
		/// \param other Shape object to compare
		/// \return true if the objects are not identical
		LIBRAPID_ALWAYS_INLINE auto operator!=(const Shape &other) const -> bool;

		/// Return the number of dimensions in the Shape object
		/// \return Number of dimensions
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto ndim() const -> int;

		/// Return a subshape of the Shape object
		/// \param start Starting index
		/// \param end Ending index
		/// \return Subshape
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto subshape(size_t start, size_t end) const
		  -> Shape;

		/// Return the number of elements the Shape object represents
		/// \return Number of elements
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto size() const -> size_t;

		template<typename T_, typename Char, typename Ctx>
		void str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const;

	protected:
		int m_dims;
		std::array<SizeType, N> m_data;
	};

	class MatrixShape {
	public:
		using SizeType						  = uint32_t;
		static constexpr size_t MaxDimensions = 2;

		MatrixShape() = default;

		template<typename Scalar, size_t Rows, size_t Cols>
		explicit MatrixShape(const FixedStorage<Scalar, Rows, Cols> &fixed);

		template<typename V>
		MatrixShape(const std::initializer_list<V> &vals);

		template<typename V>
		explicit MatrixShape(const std::vector<V> &vals);

		MatrixShape(const MatrixShape &other) = default;

		MatrixShape(MatrixShape &&other) noexcept = default;

		template<typename V>
		auto operator=(const std::initializer_list<V> &vals) -> MatrixShape &;

		template<typename V>
		auto operator=(const std::vector<V> &vals) -> MatrixShape &;

		MatrixShape &operator=(const MatrixShape &other) = default;

		MatrixShape &operator=(MatrixShape &&other) noexcept = default;

		static auto zeros() -> MatrixShape;

		static auto ones() -> MatrixShape;

		template<typename Index>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](Index index) const
		  -> const SizeType &;

		template<typename Index>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](Index index) -> SizeType &;

		LIBRAPID_ALWAYS_INLINE auto operator<=>(const MatrixShape &other) const = default;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto ndim() const -> int;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto subshape(size_t start, size_t end) const
		  -> Shape<LIBRAPID_MAX_ARRAY_DIMS>;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto size() const -> size_t;

		template<typename T_, typename Char, typename Ctx>
		void str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const;

	private:
		SizeType m_rows;
		SizeType m_cols;
	};

	namespace detail {
		template<typename T, size_t... Dims>
		Shape<LIBRAPID_MAX_ARRAY_DIMS> shapeFromFixedStorage(const FixedStorage<T, Dims...> &) {
			return Shape<LIBRAPID_MAX_ARRAY_DIMS>({Dims...});
		}
	} // namespace detail

	template<size_t N>
	template<typename Scalar, size_t... Dimensions>
	Shape<N>::Shape(const FixedStorage<Scalar, Dimensions...> &) : m_data({Dimensions...}) {}

	template<size_t N>
	template<typename V>
	Shape<N>::Shape(const std::initializer_list<V> &vals) : m_dims(vals.size()) {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		for (size_t i = 0; i < vals.size(); ++i) { m_data[i] = *(vals.begin() + i); }
	}

	template<size_t N>
	template<typename V>
	Shape<N>::Shape(const std::vector<V> &vals) : m_dims(vals.size()) {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		for (size_t i = 0; i < vals.size(); ++i) { m_data[i] = vals[i]; }
	}

	template<size_t N>
	template<size_t Dim>
	Shape<N>::Shape(const Shape<Dim> &other) : m_dims(other.ndim()) {
		LIBRAPID_ASSERT(other.ndim() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						other.ndim());
		for (size_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
	}

	template<size_t N>
	template<size_t Dim>
	Shape<N>::Shape(Shape<Dim> &&other) noexcept : m_dims(other.ndim()) {
		LIBRAPID_ASSERT(other.ndim() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						other.ndim());
		for (size_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
	}

	template<size_t N>
	template<typename V>
	auto Shape<N>::operator=(const std::initializer_list<V> &vals) -> Shape & {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		m_dims = vals.size();
		for (int64_t i = 0; i < vals.size(); ++i) { m_data[i] = *(vals.begin() + i); }
		return *this;
	}

	template<size_t N>
	template<typename V>
	auto Shape<N>::operator=(const std::vector<V> &vals) -> Shape & {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		m_dims = vals.size();
		for (int64_t i = 0; i < vals.size(); ++i) { m_data[i] = vals[i]; }
		return *this;
	}

	template<size_t N>
	auto Shape<N>::zeros(size_t dims) -> Shape {
		Shape res;
		res.m_dims = dims;
		for (size_t i = 0; i < dims; ++i) res.m_data[i] = 0;
		return res;
	}

	template<size_t N>
	auto Shape<N>::ones(size_t dims) -> Shape {
		Shape res;
		res.m_dims = dims;
		for (size_t i = 0; i < dims; ++i) res.m_data[i] = 1;
		return res;
	}

	template<size_t N>
	template<typename Index>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto Shape<N>::operator[](Index index) const
	  -> const SizeType & {
		static_assert(std::is_integral_v<Index>, "Index must be an integral type");
		LIBRAPID_ASSERT(index < m_dims, "Index out of bounds");
		LIBRAPID_ASSERT(index >= 0, "Index out of bounds");
		return m_data[index];
	}

	template<size_t N>
	template<typename Index>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto Shape<N>::operator[](Index index) -> SizeType & {
		static_assert(std::is_integral_v<Index>, "Index must be an integral type");
		LIBRAPID_ASSERT(index < m_dims, "Index out of bounds");
		LIBRAPID_ASSERT(index >= 0, "Index out of bounds");
		return m_data[index];
	}

	template<size_t N>
	LIBRAPID_ALWAYS_INLINE auto Shape<N>::operator==(const Shape &other) const -> bool {
		if (m_dims != other.m_dims) return false;
		for (size_t i = 0; i < m_dims; ++i) {
			if (m_data[i] != other.m_data[i]) return false;
		}
		return true;
	}

	template<size_t N>
	LIBRAPID_ALWAYS_INLINE auto Shape<N>::operator!=(const Shape &other) const -> bool {
		return !(*this == other);
	}

	template<size_t N>
	LIBRAPID_NODISCARD auto Shape<N>::ndim() const -> int {
		return m_dims;
	}

	template<size_t N>
	LIBRAPID_NODISCARD auto Shape<N>::subshape(size_t start, size_t end) const -> Shape {
		LIBRAPID_ASSERT(start <= end, "Start index must be less than end index");
		LIBRAPID_ASSERT(end <= m_dims,
						"End index must be less than or equal to the number of dimensions");
		LIBRAPID_ASSERT(start >= 0, "Start index must be greater than or equal to 0");

		Shape res;
		res.m_dims = end - start;
		for (size_t i = 0; i < res.m_dims; ++i) res.m_data[i] = m_data[i + start];
		return res;
	}

	template<size_t N>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto Shape<N>::size() const -> size_t {
		size_t res = 1;
		for (size_t i = 0; i < m_dims; ++i) res *= m_data[i];
		return res;
	}

	template<size_t N>
	template<typename T_, typename Char, typename Ctx>
	void Shape<N>::str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const {
		fmt::format_to(ctx.out(), "Shape(");
		for (size_t i = 0; i < m_dims; ++i) {
			format.format(m_data[i], ctx);
			if (i != m_dims - 1) fmt::format_to(ctx.out(), ", ");
		}
		fmt::format_to(ctx.out(), ")");
	}

	template<typename Scalar, size_t Rows, size_t Cols>
	MatrixShape::MatrixShape(const FixedStorage<Scalar, Rows, Cols> &) :
			m_rows(Rows), m_cols(Cols) {}

	template<typename V>
	MatrixShape::MatrixShape(const std::initializer_list<V> &vals) {
		LIBRAPID_ASSERT(vals.size() == 2, "MatrixShape must be initialized with 2 values");
		m_rows = *(vals.begin());
		m_cols = *(vals.begin() + 1);
	}

	template<typename V>
	MatrixShape::MatrixShape(const std::vector<V> &vals) {
		LIBRAPID_ASSERT(vals.size() == 2, "MatrixShape must be initialized with 2 values");
		m_rows = vals[0];
		m_cols = vals[1];
	}

	template<typename V>
	auto MatrixShape::operator=(const std::initializer_list<V> &vals) -> MatrixShape & {
		LIBRAPID_ASSERT(vals.size() == 2, "MatrixShape must be initialized with 2 values");
		m_rows = *(vals.begin());
		m_cols = *(vals.begin() + 1);
		return *this;
	}

	template<typename V>
	auto MatrixShape::operator=(const std::vector<V> &vals) -> MatrixShape & {
		LIBRAPID_ASSERT(vals.size() == 2, "MatrixShape must be initialized with 2 values");
		m_rows = vals[0];
		m_cols = vals[1];
		return *this;
	}

	LIBRAPID_ALWAYS_INLINE auto MatrixShape::zeros() -> MatrixShape { return MatrixShape({0, 0}); }

	LIBRAPID_ALWAYS_INLINE auto MatrixShape::ones() -> MatrixShape { return MatrixShape({1, 1}); }

	template<typename Index>
	auto MatrixShape::operator[](Index index) const -> const SizeType & {
		static_assert(std::is_integral_v<Index>, "Index must be an integral type");
		LIBRAPID_ASSERT(index < 2, "Index out of bounds");
		LIBRAPID_ASSERT(index >= 0, "Index out of bounds");

		return index == 0 ? m_rows : m_cols;
	}

	template<typename Index>
	auto MatrixShape::operator[](Index index) -> SizeType & {
		static_assert(std::is_integral_v<Index>, "Index must be an integral type");
		LIBRAPID_ASSERT(index < 2, "Index out of bounds");
		LIBRAPID_ASSERT(index >= 0, "Index out of bounds");

		return index == 0 ? m_rows : m_cols;
	}

	constexpr auto MatrixShape::ndim() const -> int { return 2; }

	auto MatrixShape::subshape(size_t start, size_t end) const -> Shape<LIBRAPID_MAX_ARRAY_DIMS> {
		LIBRAPID_ASSERT(start <= end, "Start index must be less than end index");
		LIBRAPID_ASSERT(end <= 2,
						"End index must be less than or equal to the number of dimensions");
		LIBRAPID_ASSERT(start >= 0, "Start index must be greater than or equal to 0");

		Shape<LIBRAPID_MAX_ARRAY_DIMS> res;
		res[0] = m_rows;
		res[1] = m_cols;
		return res.subshape(start, end);
	}

	auto MatrixShape::size() const -> size_t { return m_rows * m_cols; }

	template<typename T_, typename Char, typename Ctx>
	void MatrixShape::str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const {
		fmt::format_to(ctx.out(), "MatrixShape(");
		format.format(m_rows, ctx);
		fmt::format_to(ctx.out(), ", ");
		format.format(m_cols, ctx);
		fmt::format_to(ctx.out(), ")");
	}

	template<size_t N>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator==(const Shape<N> &lhs,
															  const MatrixShape &rhs) -> bool {
		return lhs.ndim() == 2 && lhs[0] == rhs[0] && lhs[1] == rhs[1];
	}

	template<size_t N>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator==(const MatrixShape &lhs,
															  const Shape<N> &rhs) -> bool {
		return rhs == lhs;
	}

	template<size_t N>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator!=(const Shape<N> &lhs,
															  const MatrixShape &rhs) -> bool {
		return !(lhs == rhs);
	}

	template<size_t N>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator!=(const MatrixShape &lhs,
															  const Shape<N> &rhs) -> bool {
		return !(lhs == rhs);
	}

	namespace typetraits {
		template<typename T>
		struct IsSizeType : std::false_type {};

		template<size_t N>
		struct IsSizeType<Shape<N>> : std::true_type {};

		template<>
		struct IsSizeType<MatrixShape> : std::true_type {};
	} // namespace typetraits

	/// Returns true if all inputs have the same shape
	/// \tparam T1 Type of the first input
	/// \tparam N1 Number of dimensions of the first input
	/// \tparam T2 Type of the second input
	/// \tparam N2 Number of dimensions of the second input
	/// \tparam Tn Type of the remaining (optional) inputs
	/// \tparam Nn Number of dimensions of the remaining (optional) inputs
	/// \param first First input
	/// \param second Second input
	/// \param shapes Remaining (optional) inputs
	/// \return True if all inputs have the same shape, false otherwise
	template<typename First, typename Second, typename... Rest,
			 std::enable_if_t<typetraits::IsSizeType<First>::value &&
								typetraits::IsSizeType<Second>::value &&
								(typetraits::IsSizeType<Rest>::value && ...),
							  int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_INLINE bool shapesMatch(const First &first, const Second &second,
														const Rest &...shapes) {
		if constexpr (sizeof...(Rest) == 0) {
			return first == second;
		} else {
			return first == second && shapesMatch(first, shapes...);
		}
	}

	/// \sa shapesMatch
	template<typename First, typename Second, typename... Rest,
			 std::enable_if_t<typetraits::IsSizeType<First>::value &&
								typetraits::IsSizeType<Second>::value &&
								(typetraits::IsSizeType<Rest>::value && ...),
							  int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_INLINE bool
	shapesMatch(const std::tuple<First, Second, Rest...> &shapes) {
		if constexpr (sizeof...(Rest) == 0) {
			return std::get<0>(shapes) == std::get<1>(shapes);
		} else {
			return std::get<0>(shapes) == std::get<1>(shapes) &&
				   shapesMatch(std::apply(
					 [](auto, auto, auto... rest) { return std::make_tuple(rest...); }, shapes));
		}
	}
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
template<size_t N>
struct fmt::formatter<librapid::Shape<N>> {
private:
	using Type	   = librapid::Shape<N>;
	using SizeType = librapid::Shape<N>::SizeType;
	using Base	   = fmt::formatter<SizeType, char>;
	Base m_base;

public:
	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const Type &val, FormatContext &ctx) const -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};

template<>
struct fmt::formatter<librapid::MatrixShape> {
private:
	using Type	   = librapid::MatrixShape;
	using SizeType = librapid::MatrixShape::SizeType;
	using Base	   = fmt::formatter<SizeType, char>;
	Base m_base;

public:
	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const Type &val, FormatContext &ctx) const -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};
#endif // FMT_API

#endif // LIBRAPID_ARRAY_SIZETYPE_HPP