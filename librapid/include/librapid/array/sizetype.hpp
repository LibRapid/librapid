#ifndef LIBRAPID_ARRAY_SIZETYPE_HPP
#define LIBRAPID_ARRAY_SIZETYPE_HPP

/*
 * This file defines the Shape class and some helper functions,
 * including stride operations.
 */

namespace librapid {
	template<typename T = size_t, size_t N = 32>
	class Shape {
	public:
		using SizeType						  = T;
		static constexpr size_t MaxDimensions = N;

		/// Default constructor
		Shape() = default;

		/// Create a Shape object from a list of values
		/// \tparam V Scalar type of the values
		/// \param vals The dimensions for the object
		template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value> = 0>
		Shape(const std::initializer_list<V> &vals);

		/// Create a Shape object from a vector of values
		/// \tparam V Scalar type of the values
		/// \param vals The dimensions for the object
		template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value> = 0>
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
		template<typename V, size_t Dim>
		Shape(const Shape<V, Dim> &other);

		/// Create a Shape object from one with a different type and number of dimensions, moving it
		/// instead of copying it.
		/// \tparam V Scalar type of the values
		/// \tparam Dim Number of dimensions
		/// \param other Temporary Shape object to move
		template<typename V, size_t Dim>
		Shape(Shape<V, Dim> &&other) noexcept;

		/// Assign a Shape object to this object
		/// \tparam V Scalar type of the Shape
		/// \param vals Dimensions of the Shape
		/// \return *this
		template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value> = 0>
		Shape &operator=(const std::initializer_list<V> &vals);

		/// Assign a Shape object to this object
		/// \tparam V Scalar type of the Shape
		/// \param vals Dimensions of the Shape
		/// \return *this
		template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value> = 0>
		Shape &operator=(const std::vector<V> &vals);

		/// Assign an RValue Shape to this object
		/// \param other RValue to move
		/// \return
		Shape &operator=(Shape &&other) noexcept = default;

		/// Return a Shape object with \p dims dimensions, all initialized to zero.
		/// \param dims Number of dimensions
		/// \return New Shape object
		static Shape zeros(size_t dims);

		/// Return a Shape object with \p dims dimensions, all initialized to one.
		/// \param dims Number of dimensions
		/// \return New Shape object
		static Shape ones(size_t dims);

		/// Access an element of the Shape object
		/// \tparam Index Typename of the index
		/// \param index Index to access
		/// \return The value at the index
		template<typename Index>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const T &operator[](Index index) const;

		/// Access an element of the Shape object
		/// \tparam Index Typename of the index
		/// \param index Index to access
		/// \return A reference to the value at the index
		template<typename Index>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T &operator[](Index index);

		/// Compare two Shape objects, returning true if and only if they are identical
		/// \param other Shape object to compare
		/// \return	true if the objects are identical
		LIBRAPID_ALWAYS_INLINE bool operator==(const Shape &other) const;

		/// Compare two Shape objects, returning true if and only if they are not identical
		/// \param other Shape object to compare
		/// \return true if the objects are not identical
		LIBRAPID_ALWAYS_INLINE bool operator!=(const Shape &other) const;

		/// Return the number of dimensions in the Shape object
		/// \return Number of dimensions
		LIBRAPID_NODISCARD T ndim() const;

		/// Return the number of elements the Shape object represents
		/// \return Number of elements
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T size() const;

		/// Convert a Shape object into a string representation
		/// \return A string representation of the Shape object
		LIBRAPID_NODISCARD std::string str() const;

	private:
		T m_dims = -1;
		std::array<T, N> m_data;
	};

	template<typename T, size_t N>
	template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value>>
	Shape<T, N>::Shape(const std::initializer_list<V> &vals) : m_dims(vals.size()) {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		for (size_t i = 0; i < vals.size(); ++i) { m_data[i] = *(vals.begin() + i); }
	}

	template<typename T, size_t N>
	template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value>>
	Shape<T, N>::Shape(const std::vector<V> &vals) : m_dims(vals.size()) {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		for (size_t i = 0; i < vals.size(); ++i) { m_data[i] = vals[i]; }
	}

	template<typename T, size_t N>
	template<typename V, size_t Dim>
	Shape<T, N>::Shape(const Shape<V, Dim> &other) : m_dims(other.ndim()) {
		LIBRAPID_ASSERT(other.ndim() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						other.ndim());
		for (size_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
	}

	template<typename T, size_t N>
	template<typename V, size_t Dim>
	Shape<T, N>::Shape(Shape<V, Dim> &&other) noexcept : m_dims(other.ndim()) {
		LIBRAPID_ASSERT(other.ndim() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						other.ndim());
		for (size_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
	}

	template<typename T, size_t N>
	template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value>>
	Shape<T, N> &Shape<T, N>::operator=(const std::initializer_list<V> &vals) {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		m_dims = vals.size();
		for (int64_t i = 0; i < vals.size(); ++i) { m_data[i] = *(vals.begin() + i); }
		return *this;
	}

	template<typename T, size_t N>
	template<typename V, typename typetraits::EnableIf<typetraits::CanCast<V, T>::value>>
	Shape<T, N> &Shape<T, N>::operator=(const std::vector<V> &vals) {
		LIBRAPID_ASSERT(vals.size() <= N,
						"Shape object is limited to {} dimensions. Cannot initialize with {}",
						N,
						vals.size());
		m_dims = vals.size();
		for (int64_t i = 0; i < vals.size(); ++i) { m_data[i] = vals[i]; }
		return *this;
	}

	template<typename T, size_t N>
	Shape<T, N> Shape<T, N>::zeros(size_t dims) {
		Shape res;
		res.m_dims = dims;
		for (size_t i = 0; i < dims; ++i) res.m_data[i] = 0;
		return res;
	}

	template<typename T, size_t N>
	Shape<T, N> Shape<T, N>::ones(size_t dims) {
		Shape res;
		res.m_dims = dims;
		for (size_t i = 0; i < dims; ++i) res.m_data[i] = 1;
		return res;
	}

	template<typename T, size_t N>
	template<typename Index>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const T &Shape<T, N>::operator[](Index index) const {
		return m_data[index];
	}

	template<typename T, size_t N>
	template<typename Index>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T &Shape<T, N>::operator[](Index index) {
		return m_data[index];
	}

	template<typename T, size_t N>
	LIBRAPID_ALWAYS_INLINE bool Shape<T, N>::operator==(const Shape &other) const {
		if (m_dims != other.m_dims) return false;
		for (size_t i = 0; i < m_dims; ++i) {
			if (m_data[i] != other.m_data[i]) return false;
		}
		return true;
	}

	template<typename T, size_t N>
	LIBRAPID_ALWAYS_INLINE bool Shape<T, N>::operator!=(const Shape &other) const {
		return !(*this == other);
	}

	template<typename T, size_t N>
	LIBRAPID_NODISCARD T Shape<T, N>::ndim() const {
		return m_dims;
	}

	template<typename T, size_t N>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T Shape<T, N>::size() const {
		T res = 1;
		for (size_t i = 0; i < m_dims; ++i) res *= m_data[i];
		return res;
	}

	template<typename T, size_t N>
	std::string Shape<T, N>::str() const {
		std::string result("(");
		for (size_t i = 0; i < m_dims; ++i) {
			result += fmt::format("{}", m_data[i]);
			if (i < m_dims - 1) result += std::string(", ");
		}
		return std::operator+(result, std::string(")"));
	}

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
	template<typename T1, size_t N1, typename T2, size_t N2, typename... Tn, size_t... Nn>
	LIBRAPID_NODISCARD LIBRAPID_INLINE bool shapesMatch(const Shape<T1, N1> &first,
														const Shape<T2, N2> &second,
														const Shape<Tn, Nn> &...shapes) {
		if constexpr (sizeof...(Tn) == 0) {
			return first == second;
		} else {
			return first == second && shapesMatch(first, shapes...);
		}
	}

	/// \sa shapesMatch
	template<typename T1, size_t N1, typename T2, size_t N2, typename... Tn, size_t... Nn>
	LIBRAPID_NODISCARD LIBRAPID_INLINE bool
	shapesMatch(const std::tuple<Shape<T1, N1>, Shape<T2, N2>, Shape<Tn, Nn>...> &shapes) {
		if constexpr (sizeof...(Tn) == 0) {
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
LIBRAPID_SIMPLE_IO_IMPL(typename T COMMA size_t N, librapid::Shape<T COMMA N>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_SIZETYPE_HPP