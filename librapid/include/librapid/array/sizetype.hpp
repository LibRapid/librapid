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

		/// Return the number of dimensions in the Shape object
		/// \return
		LIBRAPID_NODISCARD T ndim() const;

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
	LIBRAPID_NODISCARD T Shape<T, N>::ndim() const {
		return m_dims;
	}

	template<typename T, size_t N>
	std::string Shape<T, N>::str() const {
		std::string result("(");
		for (size_t i = 0; i < m_dims; ++i) {
			result += fmt::format("{}", m_data[i]);
			if (i < m_dims - 1) result += ", ";
		}
		return result + ")";
	}
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
LIBRAPID_SIMPLE_IO_IMPL(typename T COMMA size_t N, librapid::Shape<T COMMA N>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_SIZETYPE_HPP