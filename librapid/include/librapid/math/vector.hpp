#ifndef LIBRAPID_MATH_VECTOR_HPP
#define LIBRAPID_MATH_VECTOR_HPP

namespace librapid {
	/// The implementation for the Vector class. It is capable of representing
	/// an n-dimensional vector with any data type and storage type. By default,
	/// the storage type is a Vc Vector, but can be replaced with custom types
	/// for different functionality.
	/// \tparam Scalar The type of each element of the vector
	/// \tparam Dims The number of dimensions of the vector
	/// \tparam StorageType The type of the storage for the vector
	template<typename Scalar, int64_t Dims = 3, typename StorageType = Vc::SimdArray<Scalar, Dims>>
	class VecImpl {
	public:
		/// Default constructor
		VecImpl() = default;

		/// Create a Vector object from a StorageType object
		/// \param arr The StorageType object to construct from
		explicit VecImpl(const StorageType &arr);

		/// Construct a Vector object from a Vc Vector
		/// \tparam T The type of each element of the other vector
		/// \tparam ABI The underlying storage type of the other vector
		/// \param arr The Vc Vector object to construct from
		template<typename T, typename ABI>
		explicit VecImpl(const Vc::Vector<T, ABI> &arr);

		/// Construct a Vector from another Vector with potentially different dimensions,
		/// scalar type and storage type
		/// \tparam S The scalar type of the other vector
		/// \tparam D The number of dimensions of
		/// \tparam ST The storage type of the other vector
		/// \param other The other vector to construct from
		template<typename S, int64_t D, typename ST>
		VecImpl(const VecImpl<S, D, ST> &other);

		/// Construct a Vector object from n values, where n is the number of dimensions of the
		/// vector
		/// \tparam Args Parameter pack template type
		/// \param args The values to construct the vector from
		template<typename... Args, typename std::enable_if_t<sizeof...(Args) == Dims, int> = 0>
		VecImpl(Args... args);

		/// Construct a Vector object from an arbitrary number of arguments. See other
		/// vector constructors for more information
		/// \tparam Args Parameter pack template type
		/// \tparam size Number of arguments passed
		/// \param args Values
		template<typename... Args, int64_t size = sizeof...(Args),
				 typename std::enable_if_t<size != Dims, int> = 0>
		VecImpl(Args... args);

		/// Construct a Vector object from an std::initializer_list
		/// \tparam T The type of each element of the initializer list
		/// \param list The initializer list to construct from
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		VecImpl(const std::initializer_list<T> &list);

		/// Create a Vector from another vector instance
		/// \param other Vector to copy values from
		VecImpl(const VecImpl &other) = default;

		/// Move constructor for Vector objects
		/// \param other Vector to move
		VecImpl(VecImpl &&other) noexcept = default;

		/// Assignment operator for Vector objects
		/// \param other Vector to copy values from
		/// \return Reference to this
		VecImpl &operator=(const VecImpl &other) = default;

		/// Assignment move constructor for Vector objects
		/// \param other Vector to move
		/// \return Reference to this
		VecImpl &operator=(VecImpl &&other) noexcept = default;

		// Implement conversion to and from GLM data types
#ifdef GLM_VERSION

		/// Construct a Vector from a GLM Vector
		template<glm::qualifier p = glm::defaultp>
		VecImpl(const glm::vec<Dims, Scalar, p> &vec);

		/// Convert a GLM vector to a Vector object
		template<glm::qualifier p = glm::defaultp>
		operator glm::vec<Dims, Scalar, p>() const;

#endif // GLM_VERSION

		/// Access a specific element of the vector
		/// \param index The index of the element to access
		/// \return Reference to the element
		LIBRAPID_NODISCARD auto operator[](int64_t index) const;

		/// Access a specific element of the vector
		/// \param index The index of the element to access
		/// \return Reference to the element
		LIBRAPID_NODISCARD auto operator[](int64_t index);

		/// Add a vector to this vector, element-by-element
		/// \param other The vector to add
		/// \return Reference to this
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator+=(const VecImpl<T, d, S> &other);

		/// Subtract a vector from this vector, element-by-element
		/// \param other The vector to subtract
		/// \return Reference to this
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator-=(const VecImpl<T, d, S> &other);

		/// Multiply this vector by another vector, element-by-element
		/// \param other The vector to multiply by
		/// \return Reference to this
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator*=(const VecImpl<T, d, S> &other);

		/// Divide this vector by another vector, element-by-element
		/// \param other The vector to divide by
		/// \return Reference to this
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator/=(const VecImpl<T, d, S> &other);

		/// Add a scalar to this vector, element-by-element
		/// \param other The scalar to add
		/// \return Reference to this
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator+=(const T &value);

		/// Subtract a scalar from this vector, element-by-element
		/// \param other The scalar to subtract
		/// \return Reference to this
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator-=(const T &value);

		/// Multiply this vector by a scalar, element-by-element
		/// \param other The scalar to multiply by
		/// \return Reference to this
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator*=(const T &value);

		/// Divide this vector by a scalar, element-by-element
		/// \param other The scalar to divide by
		/// \return Reference to this
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl &operator/=(const T &value);

		/// Negate this vector
		/// \return Vector with all elements negated
		LIBRAPID_ALWAYS_INLINE VecImpl operator-() const;

		/// Compare two vectors for equality. Available modes are:
		/// - "eq" - Check for equality
		/// - "ne" - Check for inequality
		/// - "lt" - Check if each element is less than the corresponding element in the other
		/// - "le" - Check if each element is less than or equal to the corresponding element in the
		/// other
		/// - "gt" - Check if each element is greater than the corresponding element in the other
		/// - "ge" - Check if each element is greater than or equal to the corresponding element in
		/// the other \param other The vector to compare to \param mode The comparison mode \return
		/// Vector with each element set to 1 if the comparison is true, 0 otherwise
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl cmp(const VecImpl<T, d, S> &other, const char *mode) const;

		/// Compare a vector and a scalar for equality. Available modes are:
		/// - "eq" - Check for equality
		/// - "ne" - Check for inequality
		/// - "lt" - Check if each element is less than the scalar
		/// - "le" - Check if each element is less than or equal to the scalar
		/// - "gt" - Check if each element is greater than the scalar
		/// - "ge" - Check if each element is greater than or equal to the scalar
		/// \param value The scalar to compare to
		/// \param mode The comparison mode
		/// \return Vector with each element set to 1 if the comparison is true, 0 otherwise
		template<typename T>
		LIBRAPID_ALWAYS_INLINE VecImpl cmp(const T &value, const char *mode) const;

		/// Equivalent to calling cmp(other, "lt")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl operator<(const VecImpl<T, d, S> &other) const;

		/// Equivalent to calling cmp(other, "le")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl operator<=(const VecImpl<T, d, S> &other) const;

		/// Equivalent to calling cmp(other, "gt")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl operator>(const VecImpl<T, d, S> &other) const;

		/// Equivalent to calling cmp(other, "ge")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl operator>=(const VecImpl<T, d, S> &other) const;

		/// Equivalent to calling cmp(other, "eq")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl operator==(const VecImpl<T, d, S> &other) const;

		/// Equivalent to calling cmp(other, "ne")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d, typename S>
		LIBRAPID_ALWAYS_INLINE VecImpl operator!=(const VecImpl<T, d, S> &other) const;

		/// Equivalent to calling cmp(other, "lt")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl operator<(const T &other) const;

		/// Equivalent to calling cmp(other, "le")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl operator<=(const T &other) const;

		/// Equivalent to calling cmp(other, "gt")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl operator>(const T &other) const;

		/// Equivalent to calling cmp(other, "ge")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl operator>=(const T &other) const;

		/// Equivalent to calling cmp(other, "eq")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl operator==(const T &other) const;

		/// Equivalent to calling cmp(other, "ne")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE VecImpl operator!=(const T &other) const;

		/// Calculate the magnitude of this vector squared
		/// \return The magnitude squared
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar mag2() const;

		/// Calculate the magnitude of this vector
		/// \return The magnitude
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar mag() const;

		/// Calculate 1/mag(this)
		/// \return 1/mag(this)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar invMag() const { return 1.0 / mag(); }

		/// Calculate the normalized version of this vector
		/// \return The normalized vector
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE VecImpl norm() const;

		/// Calculate the dot product of this vector and another
		/// \param other The other vector
		/// \return The dot product
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar dot(const VecImpl &other) const;

		/// Calculate the cross product of this vector and another
		/// \param other The other vector
		/// \return The cross product
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE VecImpl cross(const VecImpl &other) const;

		/// Project this vector onto another vector
		/// \param other The vector to project onto
		/// \return The projection of this vector onto the other vector
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE VecImpl proj(const VecImpl &other) const;

		/// Cast this vector to a boolean. This is equivalent to calling mag2() != 0
		/// \return True if the magnitude of this vector is not 0, false otherwise
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE explicit operator bool() const;

		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 2, StorageType> xy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 2, StorageType> yx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 2, StorageType> xz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 2, StorageType> zx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 2, StorageType> yz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 2, StorageType> zy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> xyz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> xzy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> yxz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> yzx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> zxy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> zyx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> xyw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> xwy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> yxw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> ywx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> wxy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> wyx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> xzw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> xwz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> zxw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> zwx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> wxz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> wzx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> yzw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> ywz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> zyw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> zwy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> wyz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 3, StorageType> wzy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> xyzw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> xywz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> xzyw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> xzwy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> xwyz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> xwzy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> yxzw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> yxwz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> yzxw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> yzwx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> ywxz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> ywzx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> zxyw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> zxwy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> zyxw() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> zywx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> zwxy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> zwyx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> wxyz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> wxzy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> wyxz() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> wyzx() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> wzxy() const;
		LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, 4, StorageType> wzyx() const;

		/// Return the underlying storage type
		/// \return The underlying storage type
		LIBRAPID_ALWAYS_INLINE const StorageType &data() const;

		/// Return the underlying storage type
		/// \return The underlying storage type
		LIBRAPID_ALWAYS_INLINE StorageType &data();

		/// Access the x component of this vector
		/// \return The x component of this vector
		LIBRAPID_ALWAYS_INLINE Scalar x() const;

		/// Access the y component of this vector
		/// \return The y component of this vector
		LIBRAPID_ALWAYS_INLINE Scalar y() const;

		/// Access the z component of this vector
		/// \return The z component of this vector
		LIBRAPID_ALWAYS_INLINE Scalar z() const;

		/// Access the w component of this vector
		/// \return The w component of this vector
		LIBRAPID_ALWAYS_INLINE Scalar w() const;

		/// Set the x component of this vector
		/// \param val The new value of the x component
		LIBRAPID_ALWAYS_INLINE void x(Scalar val);

		/// Set the y component of this vector
		/// \param val The new value of the y component
		LIBRAPID_ALWAYS_INLINE void y(Scalar val);

		/// Set the z component of this vector
		/// \param val The new value of the z component
		LIBRAPID_ALWAYS_INLINE void z(Scalar val);

		/// Set the w component of this vector
		/// \param val The new value of the w component
		LIBRAPID_ALWAYS_INLINE void w(Scalar val);

		/// Convert a vector into a string representation -- "(x, y, z, w, ...)"
		/// \param formatString The format string to use for each component
		/// \return A string representation of this vector
		LIBRAPID_NODISCARD std::string str(const std::string &formatString = "{}") const;

	protected:
		StorageType m_data {};
	};

	template<typename Scalar, int64_t Dims, typename StorageType>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(const StorageType &arr) : m_data {arr} {}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename ABI>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(const Vc::Vector<T, ABI> &arr) : m_data {arr} {}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename S, int64_t D, typename ST>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(const VecImpl<S, D, ST> &other) {
		for (int64_t i = 0; i < min(Dims, D); ++i) {
			m_data[i] = static_cast<Scalar>(other[i]);
		}
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename... Args, typename std::enable_if_t<sizeof...(Args) == Dims, int>>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(Args... args) :
			m_data {static_cast<Scalar>(args)...} {
		static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename... Args, int64_t size, typename std::enable_if_t<size != Dims, int>>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(Args... args) {
		static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
		if constexpr (size == 1) {
			m_data = StorageType(static_cast<Scalar>(args)...);
		} else {
			const Scalar expanded[] = {static_cast<Scalar>(args)...};
			for (int64_t i = 0; i < size; i++) { m_data[i] = expanded[i]; }
		}
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(const std::initializer_list<T> &list) {
		assert(list.size() <= Dims);
		int64_t i = 0;
		for (const Scalar &val : list) { m_data[i++] = val; }
	}

	template<typename Scalar, int64_t Dims, typename StorageType1, typename StorageType2>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType1>
	operator&(const VecImpl<Scalar, Dims, StorageType1> &vec,
			  const VecImpl<Scalar, Dims, StorageType2> &mask) {
		VecImpl<Scalar, Dims, StorageType1> res(vec);
		for (int64_t i = 0; i < Dims; ++i)
			if (!mask[i]) res[i] = 0;
		return res;
	}

#ifdef GLM_VERSION

	/// Construct a Vector from a GLM Vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	template<glm::qualifier p>
	VecImpl<Scalar, Dims, StorageType>::VecImpl(const glm::vec<Dims, Scalar, p> &vec) {
		for (int64_t i = 0; i < Dims; ++i) { m_data[i] = vec[i]; }
	}

	/// Convert a GLM vector to a Vector object
	template<typename Scalar, int64_t Dims, typename StorageType>
	template<glm::qualifier p>
	VecImpl<Scalar, Dims, StorageType>::operator glm::vec<Dims, Scalar, p>() const {
		glm::vec<Dims, Scalar, p> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = m_data[i]; }
		return res;
	}

#endif // GLM_VERSION

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::operator[](int64_t index) const {
		LIBRAPID_ASSERT(
		  0 <= index < Dims, "Index {} out of range for vector with {} dimensions", index, Dims);
		return m_data[index];
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::operator[](int64_t index) {
		LIBRAPID_ASSERT(
		  0 <= index < Dims, "Index {} out of range for vector with {} dimensions", index, Dims);
		return m_data[index];
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator+=(const VecImpl<T, d, S> &other)
	  -> VecImpl & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data += other.m_data;
		} else {
			for (int64_t i = 0; i < d; ++i) { m_data[i] += other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator-=(const VecImpl<T, d, S> &other)
	  -> VecImpl & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data -= other.m_data;
		} else {
			for (int64_t i = 0; i < d; ++i) { m_data[i] -= other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator*=(const VecImpl<T, d, S> &other)
	  -> VecImpl & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data *= other.m_data;
		} else {
			for (int64_t i = 0; i < d; ++i) { m_data[i] *= other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator/=(const VecImpl<T, d, S> &other)
	  -> VecImpl & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data /= other.m_data;
		} else {
			for (int64_t i = 0; i < d; ++i) { m_data[i] /= other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator+=(const T &value) -> VecImpl & {
		m_data += static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator-=(const T &value) -> VecImpl & {
		m_data -= static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator*=(const T &value) -> VecImpl & {
		m_data *= static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator/=(const T &value) -> VecImpl & {
		m_data /= static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::operator-() const -> VecImpl {
		VecImpl res(*this);
		res *= -1;
		return res;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::cmp(const VecImpl<T, d, S> &other,
												 const char *mode) const -> VecImpl {
		VecImpl res(*this);
		int16_t modeInt = *(int16_t *)mode;
		for (int64_t i = 0; i < Dims; ++i) {
			switch (modeInt) {
				case 'e' | ('q' << 8):
					if (res[i] == other[i])
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'n' | ('e' << 8):
					if (res[i] != other[i])
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'l' | ('t' << 8):
					if (res[i] < other[i])
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'l' | ('e' << 8):
					if (res[i] <= other[i])
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'g' | ('t' << 8):
					if (res[i] > other[i])
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'g' | ('e' << 8):
					if (res[i] >= other[i])
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				default: LIBRAPID_ASSERT(false, "Invalid mode {}", mode);
			}
		}
		return res;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T>
	auto VecImpl<Scalar, Dims, StorageType>::cmp(const T &value, const char *mode) const
	  -> VecImpl {
		// Mode:
		// 0: ==    1: !=
		// 2: <     3: <=
		// 4: >     5: >=

		VecImpl res(*this);
		int16_t modeInt = *(int16_t *)mode;
		for (int64_t i = 0; i < Dims; ++i) {
			switch (modeInt) {
				case 'e' | ('q' << 8):
					if (res[i] == Scalar(value))
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'n' | ('e' << 8):
					if (res[i] != Scalar(value))
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'l' | ('t' << 8):
					if (res[i] < Scalar(value))
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'l' | ('e' << 8):
					if (res[i] <= Scalar(value))
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'g' | ('t' << 8):
					if (res[i] > Scalar(value))
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				case 'g' | ('e' << 8):
					if (res[i] >= Scalar(value))
						res[i] = Scalar(1);
					else
						res[i] = Scalar(0);
					break;
				default: LIBRAPID_ASSERT(false, "Invalid mode {}", mode);
			}
		}
		return res;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator<(const VecImpl<T, d, S> &other) const
	  -> VecImpl {
		return cmp(other, "lt");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator<=(const VecImpl<T, d, S> &other) const
	  -> VecImpl {
		return cmp(other, "le");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator>(const VecImpl<T, d, S> &other) const
	  -> VecImpl {
		return cmp(other, "gt");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator>=(const VecImpl<T, d, S> &other) const
	  -> VecImpl {
		return cmp(other, "ge");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator==(const VecImpl<T, d, S> &other) const
	  -> VecImpl {
		return cmp(other, "eq");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, int64_t d, typename S>
	auto VecImpl<Scalar, Dims, StorageType>::operator!=(const VecImpl<T, d, S> &other) const
	  -> VecImpl {
		return cmp(other, "ne");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator<(const T &other) const -> VecImpl {
		return cmp(other, "lt");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator<=(const T &other) const -> VecImpl {
		return cmp(other, "le");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator>(const T &other) const -> VecImpl {
		return cmp(other, "gt");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator>=(const T &other) const -> VecImpl {
		return cmp(other, "ge");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator==(const T &other) const -> VecImpl {
		return cmp(other, "eq");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	template<typename T, typename std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto VecImpl<Scalar, Dims, StorageType>::operator!=(const T &other) const -> VecImpl {
		return cmp(other, "ne");
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::mag2() const -> Scalar {
		return (m_data * m_data).sum();
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::mag() const -> Scalar {
		return sqrt(mag2());
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::norm() const -> VecImpl {
		VecImpl res(*this);
		res /= mag();
		return res;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::dot(const VecImpl &other) const -> Scalar {
		return (m_data * other.m_data).sum();
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::cross(const VecImpl &other) const -> VecImpl {
		static_assert(Dims == 3, "Cross product is only defined for 3D Vectors");
		return VecImpl(y() * other.z() - z() * other.y(),
					   z() * other.x() - x() * other.z(),
					   x() * other.y() - y() * other.x());
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::proj(const VecImpl &other) const -> VecImpl {
		return other * (dot(other) / other.mag2());
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	VecImpl<Scalar, Dims, StorageType>::operator bool() const {
		for (int64_t i = 0; i < Dims; ++i)
			if (m_data[i] != 0) return true;
		return false;
	}

	/// Add two Vector objects together and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res += rhs;
		return res;
	}

	/// Subtract two Vector objects and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the subtraction
	/// \param rhs The right hand side of the subtraction
	/// \return The result of the subtraction
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res -= rhs;
		return res;
	}

	/// Multiply two Vector objects element-by-element and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the multiplication
	/// \param rhs The right hand side of the multiplication
	/// \return The result of the multiplication
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res *= rhs;
		return res;
	}

	/// Divide two Vector objects element-by-element and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the division
	/// \param rhs The right hand side of the division
	/// \return The result of the division
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res /= rhs;
		return res;
	}

	/// Add a scalar to a Vector object and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename Scalar, int64_t Dims, typename StorageType, typename S>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res += rhs;
		return res;
	}

	/// Subtract a scalar from a Vector object and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the subtraction
	/// \param rhs The right hand side of the subtraction
	/// \return The result of the subtraction
	template<typename Scalar, int64_t Dims, typename StorageType, typename S>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res -= rhs;
		return res;
	}

	/// Multiply a Vector object by a scalar and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the multiplication
	/// \param rhs The right hand side of the multiplication
	/// \return The result of the multiplication
	template<typename Scalar, int64_t Dims, typename StorageType, typename S>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res *= rhs;
		return res;
	}

	/// Divide a Vector object by a scalar and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the division
	/// \param rhs The right hand side of the division
	/// \return The result of the division
	template<typename Scalar, int64_t Dims, typename StorageType, typename S>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res /= rhs;
		return res;
	}

	/// Add a scalar to a Vector object and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename S, typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res += rhs;
		return res;
	}

	/// Subtract a Vector object from a scalar and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the subtraction
	/// \param rhs The right hand side of the subtraction
	/// \return The result of the subtraction
	template<typename S, typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res -= rhs;
		return res;
	}

	/// Multiply a scalar by a Vector object and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the multiplication
	/// \param rhs The right hand side of the multiplication
	/// \return The result of the multiplication
	template<typename S, typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res *= rhs;
		return res;
	}

	/// Divide a scalar by a Vector object and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the division
	/// \param rhs The right hand side of the division
	/// \return The result of the division
	template<typename S, typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res /= rhs;
		return res;
	}

	/// Add two Vector objects and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam StorageType The type of the storage
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xy() const -> VecImpl<Scalar, 2, StorageType> {
		return {x(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yx() const -> VecImpl<Scalar, 2, StorageType> {
		return {y(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xz() const -> VecImpl<Scalar, 2, StorageType> {
		return {x(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zx() const -> VecImpl<Scalar, 2, StorageType> {
		return {z(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yz() const -> VecImpl<Scalar, 2, StorageType> {
		return {y(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zy() const -> VecImpl<Scalar, 2, StorageType> {
		return {z(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xyz() const -> VecImpl<Scalar, 3, StorageType> {
		return {x(), y(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xzy() const -> VecImpl<Scalar, 3, StorageType> {
		return {x(), z(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yxz() const -> VecImpl<Scalar, 3, StorageType> {
		return {y(), x(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yzx() const -> VecImpl<Scalar, 3, StorageType> {
		return {y(), z(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zxy() const -> VecImpl<Scalar, 3, StorageType> {
		return {z(), x(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zyx() const -> VecImpl<Scalar, 3, StorageType> {
		return {z(), y(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xyw() const -> VecImpl<Scalar, 3, StorageType> {
		return {x(), y(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xwy() const -> VecImpl<Scalar, 3, StorageType> {
		return {x(), w(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yxw() const -> VecImpl<Scalar, 3, StorageType> {
		return {y(), x(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::ywx() const -> VecImpl<Scalar, 3, StorageType> {
		return {y(), w(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wxy() const -> VecImpl<Scalar, 3, StorageType> {
		return {w(), x(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wyx() const -> VecImpl<Scalar, 3, StorageType> {
		return {w(), y(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xzw() const -> VecImpl<Scalar, 3, StorageType> {
		return {x(), z(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xwz() const -> VecImpl<Scalar, 3, StorageType> {
		return {x(), w(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zxw() const -> VecImpl<Scalar, 3, StorageType> {
		return {z(), x(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zwx() const -> VecImpl<Scalar, 3, StorageType> {
		return {z(), w(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wxz() const -> VecImpl<Scalar, 3, StorageType> {
		return {w(), x(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wzx() const -> VecImpl<Scalar, 3, StorageType> {
		return {w(), z(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yzw() const -> VecImpl<Scalar, 3, StorageType> {
		return {y(), z(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::ywz() const -> VecImpl<Scalar, 3, StorageType> {
		return {y(), w(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zyw() const -> VecImpl<Scalar, 3, StorageType> {
		return {z(), y(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zwy() const -> VecImpl<Scalar, 3, StorageType> {
		return {z(), w(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wyz() const -> VecImpl<Scalar, 3, StorageType> {
		return {w(), y(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wzy() const -> VecImpl<Scalar, 3, StorageType> {
		return {w(), z(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xyzw() const -> VecImpl<Scalar, 4, StorageType> {
		return {x(), y(), z(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xywz() const -> VecImpl<Scalar, 4, StorageType> {
		return {x(), y(), w(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xzyw() const -> VecImpl<Scalar, 4, StorageType> {
		return {x(), z(), y(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xzwy() const -> VecImpl<Scalar, 4, StorageType> {
		return {x(), z(), w(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xwyz() const -> VecImpl<Scalar, 4, StorageType> {
		return {x(), w(), y(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::xwzy() const -> VecImpl<Scalar, 4, StorageType> {
		return {x(), w(), z(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yxzw() const -> VecImpl<Scalar, 4, StorageType> {
		return {y(), x(), z(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yxwz() const -> VecImpl<Scalar, 4, StorageType> {
		return {y(), x(), w(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yzxw() const -> VecImpl<Scalar, 4, StorageType> {
		return {y(), z(), x(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::yzwx() const -> VecImpl<Scalar, 4, StorageType> {
		return {y(), z(), w(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::ywxz() const -> VecImpl<Scalar, 4, StorageType> {
		return {y(), w(), x(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::ywzx() const -> VecImpl<Scalar, 4, StorageType> {
		return {y(), w(), z(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zxyw() const -> VecImpl<Scalar, 4, StorageType> {
		return {z(), x(), y(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zxwy() const -> VecImpl<Scalar, 4, StorageType> {
		return {z(), x(), w(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zyxw() const -> VecImpl<Scalar, 4, StorageType> {
		return {z(), y(), x(), w()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zywx() const -> VecImpl<Scalar, 4, StorageType> {
		return {z(), y(), w(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zwxy() const -> VecImpl<Scalar, 4, StorageType> {
		return {z(), w(), x(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::zwyx() const -> VecImpl<Scalar, 4, StorageType> {
		return {z(), w(), y(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wxyz() const -> VecImpl<Scalar, 4, StorageType> {
		return {w(), x(), y(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wxzy() const -> VecImpl<Scalar, 4, StorageType> {
		return {w(), x(), z(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wyxz() const -> VecImpl<Scalar, 4, StorageType> {
		return {w(), y(), x(), z()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wyzx() const -> VecImpl<Scalar, 4, StorageType> {
		return {w(), y(), z(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wzxy() const -> VecImpl<Scalar, 4, StorageType> {
		return {w(), z(), x(), y()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::wzyx() const -> VecImpl<Scalar, 4, StorageType> {
		return {w(), z(), y(), x()};
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::data() const -> const StorageType & {
		return m_data;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::data() -> StorageType & {
		return m_data;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::x() const -> Scalar {
		if constexpr (Dims < 1)
			return 0;
		else
			return m_data[0];
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::y() const -> Scalar {
		if constexpr (Dims < 2)
			return 0;
		else
			return m_data[1];
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::z() const -> Scalar {
		if constexpr (Dims < 3)
			return 0;
		else
			return m_data[2];
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::w() const -> Scalar {
		if constexpr (Dims < 4)
			return 0;
		else
			return m_data[3];
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	void VecImpl<Scalar, Dims, StorageType>::x(Scalar val) {
		if constexpr (Dims >= 1) m_data[0] = val;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	void VecImpl<Scalar, Dims, StorageType>::y(Scalar val) {
		if constexpr (Dims >= 2) m_data[1] = val;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	void VecImpl<Scalar, Dims, StorageType>::z(Scalar val) {
		if constexpr (Dims >= 3) m_data[2] = val;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	void VecImpl<Scalar, Dims, StorageType>::w(Scalar val) {
		if constexpr (Dims >= 4) m_data[3] = val;
	}

	template<typename Scalar, int64_t Dims, typename StorageType>
	auto VecImpl<Scalar, Dims, StorageType>::str(const std::string &formatString) const
	  -> std::string {
		std::string res = "(";
		for (int64_t i = 0; i < Dims; ++i) {
			res += fmt::format(formatString, m_data[i]);
			if (i < Dims - 1) res += ", ";
		}
		return res + ")";
	}

	/// Calculate the squared distance between two vectors
	/// \tparam Scalar The scalar type of the vectors
	/// \tparam Dims The dimensionality of the vectors
	/// \tparam StorageType The storage type of the vectors
	/// \param lhs The first vector
	/// \param rhs The second vector
	/// \return The squared distance between the two vectors
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE Scalar dist2(const VecImpl<Scalar, Dims, StorageType> &lhs,
										const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return (lhs - rhs).mag2();
	}

	/// Calculate the distance between two vectors
	/// \tparam Scalar The scalar type of the vectors
	/// \tparam Dims The dimensionality of the vectors
	/// \tparam StorageType The storage type of the vectors
	/// \param lhs The first vector
	/// \param rhs The second vector
	/// \return The distance between the two vectors
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE Scalar dist(const VecImpl<Scalar, Dims, StorageType> &lhs,
									   const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return (lhs - rhs).mag();
	}

	/// Calculate the sin of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the sin of
	/// \return The sin of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	sin(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::sin(vec.data()));
	}

	/// Calculate the cos of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the cos of
	/// \return The cos of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	cos(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::cos(vec.data()));
	}

	/// Calculate the tan of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the tan of
	/// \return The tan of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	tan(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(sin(vec) / cos(vec));
	}

	/// Calculate the asin of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the asin of
	/// \return The asin of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	asin(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::asin(vec.data()));
	}

	/// Calculate the acos of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the acos of
	/// \return The acos of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	acos(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(HALFPI - asin(vec));
	}

	/// Calculate the atan of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the atan of
	/// \return The atan of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	atan(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::atan(vec.data()));
	}

	/// Calculate the atan2 of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the atan2 of
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	atan2(const VecImpl<Scalar, Dims, StorageType> &lhs,
		  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::atan2(lhs.data(), rhs.data()));
	}

	/// Calculate the sinh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the sinh of
	/// \return The sinh of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	sinh(const VecImpl<Scalar, Dims, StorageType> &vec) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = std::sinh(vec[i]); }
		return res;
	}

	/// Calculate the cosh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the cosh of
	/// \return The cosh of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	cosh(const VecImpl<Scalar, Dims, StorageType> &vec) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = std::cosh(vec[i]); }
		return res;
	}

	/// Calculate the tanh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the tanh of
	/// \return The tanh of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	tanh(const VecImpl<Scalar, Dims, StorageType> &vec) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = std::tanh(vec[i]); }
		return res;
	}

	/// Calculate the asinh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the asinh of
	/// \return The asinh of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	asinh(const VecImpl<Scalar, Dims, StorageType> &vec) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = std::asinh(vec[i]); }
		return res;
	}

	/// Calculate the acosh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the acosh of
	/// \return The acosh of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	acosh(const VecImpl<Scalar, Dims, StorageType> &vec) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = std::acosh(vec[i]); }
		return res;
	}

	/// Calculate the atanh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the atanh of
	/// \return The atanh of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	atanh(const VecImpl<Scalar, Dims, StorageType> &vec) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = std::atanh(vec[i]); }
		return res;
	}

	/// Calculate the exp of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the exp of
	/// \return The exp of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	exp(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::exp(vec.data()));
	}

	/// Calculate the log of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the log of
	/// \return The log of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	log(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::log(vec.data()));
	}

	/// Calculate the log10 of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the log10 of
	/// \return The log10 of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	log10(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::log10(vec.data()));
	}

	/// Calculate the log2 of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the log2 of
	/// \return The log2 of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	log2(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::log2(vec.data()));
	}

	/// Calculate the sqrt of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the sqrt of
	/// \return The sqrt of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	sqrt(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::sqrt(vec.data()));
	}

	/// Raise each element of a vector to the power of another vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec Base vector
	/// \param exp Vector of exponents
	/// \return The result of raising each element of the vector to the power of the corresponding
	///         element of the exponent vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	pow(const VecImpl<Scalar, Dims, StorageType> &vec,
		const VecImpl<Scalar, Dims, StorageType> &exp) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = pow(vec[i], exp[i]); }
		return res;
	}

	/// Raise each element of a vector to the power of a scalar and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \tparam T The scalar type of the exponent
	/// \param vec Base vector
	/// \param exp Scalar exponent
	/// \return The result of raising each element of the vector to the power of the scalar
	template<typename Scalar, int64_t Dims, typename StorageType, typename T>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	pow(const VecImpl<Scalar, Dims, StorageType> &vec, T exp) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) {
			res[i] = static_cast<Scalar>(pow(vec[i], static_cast<Scalar>(exp)));
		}
		return res;
	}

	/// Raise a scalar to the power of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec Base vector
	/// \param exp Scalar exponent
	/// \return The result of raising the scalar to the power of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	pow(Scalar vec, const VecImpl<Scalar, Dims, StorageType> &exp) {
		VecImpl<Scalar, Dims, StorageType> res;
		for (int64_t i = 0; i < Dims; ++i) { res[i] = pow(vec, exp[i]); }
		return res;
	}

	/// Calculate the cbrt of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the cbrt of
	/// \return The cbrt of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	cbrt(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return pow(vec, 1.0 / 3.0);
	}

	/// Calculate the absolute value of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the absolute value of
	/// \return The absolute value of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	abs(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::abs(vec.data()));
	}

	/// Calculate the floor of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the floor of
	/// \return The floor of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	floor(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::floor(vec.data()));
	}

	/// Calculate the ceil of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam StorageType The storage type of the vector
	/// \param vec The vector to calculate the ceil of
	/// \return The ceil of each element of the vector
	template<typename Scalar, int64_t Dims, typename StorageType>
	LIBRAPID_ALWAYS_INLINE VecImpl<Scalar, Dims, StorageType>
	ceil(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::ceil(vec.data()));
	}

	/// A simplified interface to the VecImpl class, defaulting to Vc SimdArray storage
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	template<typename Scalar, int64_t Dims>
	using Vec = VecImpl<Scalar, Dims, Vc::SimdArray<Scalar, Dims>>;

	using Vec2i = Vec<int32_t, 2>;
	using Vec3i = Vec<int32_t, 3>;
	using Vec4i = Vec<int32_t, 4>;
	using Vec2f = Vec<float, 2>;
	using Vec3f = Vec<float, 3>;
	using Vec4f = Vec<float, 4>;
	using Vec2d = Vec<double, 2>;
	using Vec3d = Vec<double, 3>;
	using Vec4d = Vec<double, 4>;

	using Vec2 = Vec2d;
	using Vec3 = Vec3d;
	using Vec4 = Vec4d;

	template<typename Scalar, int64_t Dims, typename StorageType>
	std::ostream &operator<<(std::ostream &os, const VecImpl<Scalar, Dims, StorageType> &vec) {
		os << vec.str();
		return os;
	}
} // namespace librapid

#ifdef FMT_API
template<typename Scalar, int64_t D, typename StorageType>
struct fmt::formatter<librapid::VecImpl<Scalar, D, StorageType>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::VecImpl<Scalar, D, StorageType> &arr, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), arr.str());
	}
};
#endif // FMT_API

#endif // LIBRAPID_MATH_VECTOR_HPP
