#ifndef LIBRAPID_MATH_SIMD_VECTOR_HPP
#define LIBRAPID_MATH_SIMD_VECTOR_HPP

namespace librapid {
	/// The implementation for the Vector class. It is capable of representing
	/// an n-dimensional vector with any data type and storage type. By default,
	/// the storage type is a Vc Vector, but can be replaced with custom types
	/// for different functionality.
	/// \tparam Scalar The type of each element of the vector
	/// \tparam Dims The number of dimensions of the vector
	/// \tparam StorageType The type of the storage for the vector
	template<typename Scalar, int64_t Dims = 3>
	class SIMDVector {
	public:
		using StorageType = Vc::SimdArray<Scalar, Dims>;

		/// Default constructor
		SIMDVector() = default;

		/// Create a Vector object from a StorageType object
		/// \param arr The StorageType object to construct from
		explicit SIMDVector(const StorageType &arr);

		/// Construct a Vector object from a Vc Vector
		/// \tparam T The type of each element of the other vector
		/// \tparam ABI The underlying storage type of the other vector
		/// \param arr The Vc Vector object to construct from
		template<typename T, typename ABI>
		explicit SIMDVector(const Vc::Vector<T, ABI> &arr);

		/// Construct a Vector from another Vector with potentially different dimensions,
		/// scalar type and storage type
		/// \tparam S The scalar type of the other vector
		/// \tparam D The number of dimensions of
		/// \param other The other vector to construct from
		template<typename S, int64_t D>
		SIMDVector(const SIMDVector<S, D> &other);

		/// Construct a Vector object from n values, where n is the number of dimensions of the
		/// vector
		/// \tparam Args Parameter pack template type
		/// \param args The values to construct the vector from
		template<typename... Args, std::enable_if_t<sizeof...(Args) == Dims, int> = 0>
		SIMDVector(Args... args);

		/// Construct a Vector object from an arbitrary number of arguments. See other
		/// vector constructors for more information
		/// \tparam Args Parameter pack template type
		/// \tparam size Number of arguments passed
		/// \param args Values
		template<typename... Args, int64_t size = sizeof...(Args),
				 typename std::enable_if_t<size != Dims, int> = 0>
		SIMDVector(Args... args);

		/// Construct a Vector object from an std::initializer_list
		/// \tparam T The type of each element of the initializer list
		/// \param list The initializer list to construct from
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		SIMDVector(const std::initializer_list<T> &list);

		/// Create a Vector from another vector instance
		/// \param other Vector to copy values from
		SIMDVector(const SIMDVector &other) = default;

		/// Move constructor for Vector objects
		/// \param other Vector to move
		SIMDVector(SIMDVector &&other) noexcept = default;

		/// Assignment operator for Vector objects
		/// \param other Vector to copy values from
		/// \return Reference to this
		SIMDVector &operator=(const SIMDVector &other) = default;

		/// Assignment move constructor for Vector objects
		/// \param other Vector to move
		/// \return Reference to this
		SIMDVector &operator=(SIMDVector &&other) noexcept = default;

		// Implement conversion to and from GLM data types
#ifdef GLM_VERSION

		/// Construct a Vector from a GLM Vector
		template<glm::qualifier p = glm::defaultp>
		SIMDVector(const glm::vec<Dimscalar, p> &vec);

		/// Convert a GLM vector to a Vector object
		template<glm::qualifier p = glm::defaultp>
		operator glm::vec<Dimscalar, p>() const;

#endif // GLM_VERSION

		/// Access a specific element of the vector
		/// \param index The index of the element to access
		/// \return Reference to the element
		LIBRAPID_NODISCARD const auto operator[](int64_t index) const;

		/// Access a specific element of the vector
		/// \param index The index of the element to access
		/// \return Reference to the element
		LIBRAPID_NODISCARD auto operator[](int64_t index);

		/// Subtract a vector from this vector, element-by-element
		/// \param other The vector to subtract
		/// \return Reference to this
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator-=(const SIMDVector<T, d> &other);

		/// Add a vector to this vector, element-by-element
		/// \param other The vector to add
		/// \return Reference to this
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator+=(const SIMDVector<T, d> &other);

		/// Multiply this vector by another vector, element-by-element
		/// \param other The vector to multiply by
		/// \return Reference to this
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator*=(const SIMDVector<T, d> &other);

		/// Divide this vector by another vector, element-by-element
		/// \param other The vector to divide by
		/// \return Reference to this
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator/=(const SIMDVector<T, d> &other);

		/// Add a scalar to this vector, element-by-element
		/// \param other The scalar to add
		/// \return Reference to this
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator+=(const T &value);

		/// Subtract a scalar from this vector, element-by-element
		/// \param other The scalar to subtract
		/// \return Reference to this
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator-=(const T &value);

		/// Multiply this vector by a scalar, element-by-element
		/// \param other The scalar to multiply by
		/// \return Reference to this
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator*=(const T &value);

		/// Divide this vector by a scalar, element-by-element
		/// \param other The scalar to divide by
		/// \return Reference to this
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector &operator/=(const T &value);

		/// Negate this vector
		/// \return Vector with all elements negated
		LIBRAPID_ALWAYS_INLINE SIMDVector operator-() const;

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
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector cmp(const SIMDVector<T, d> &other,
											  const char *mode) const;

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
		LIBRAPID_ALWAYS_INLINE SIMDVector cmp(const T &value, const char *mode) const;

		/// Equivalent to calling cmp(other, "lt")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator<(const SIMDVector<T, d> &other) const;

		/// Equivalent to calling cmp(other, "le")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator<=(const SIMDVector<T, d> &other) const;

		/// Equivalent to calling cmp(other, "gt")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator>(const SIMDVector<T, d> &other) const;

		/// Equivalent to calling cmp(other, "ge")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator>=(const SIMDVector<T, d> &other) const;

		/// Equivalent to calling cmp(other, "eq")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator==(const SIMDVector<T, d> &other) const;

		/// Equivalent to calling cmp(other, "ne")
		/// \param other The vector to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, int64_t d>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator!=(const SIMDVector<T, d> &other) const;

		/// Equivalent to calling cmp(other, "lt")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator<(const T &other) const;

		/// Equivalent to calling cmp(other, "le")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator<=(const T &other) const;

		/// Equivalent to calling cmp(other, "gt")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator>(const T &other) const;

		/// Equivalent to calling cmp(other, "ge")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator>=(const T &other) const;

		/// Equivalent to calling cmp(other, "eq")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator==(const T &other) const;

		/// Equivalent to calling cmp(other, "ne")
		/// \param value The scalar to compare to
		/// \return See cmp()
		/// \see cmp()
		template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int> = 0>
		LIBRAPID_ALWAYS_INLINE SIMDVector operator!=(const T &other) const;

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
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector norm() const;

		/// Calculate the dot product of this vector and another
		/// \param other The other vector
		/// \return The dot product
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar dot(const SIMDVector &other) const;

		/// Calculate the cross product of this vector and another
		/// \param other The other vector
		/// \return The cross product
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector cross(const SIMDVector &other) const;

		/// \brief Project vector \p other onto this vector and return the result
		///
		/// Perform vector projection using the formula:
		/// \f$ \operatorname{proj}_a(\vec{b})=\frac{\vec{b} \cdot \vec{a}}{|\vec{a}|^2} \cdot
		/// \vec{a} \f$
		///
		/// \param other The vector to project
		/// \return The projection of \p other onto this vector
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector proj(const SIMDVector &other) const;

		/// Cast this vector to a boolean. This is equivalent to calling mag2() != 0
		/// \return True if the magnitude of this vector is not 0, false otherwise
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE explicit operator bool() const;

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

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 2> xy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 2> yx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 2> xz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 2> zx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 2> yz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 2> zy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> xyz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> xzy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> yxz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> yzx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> zxy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> zyx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> xyw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> xwy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> yxw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> ywx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> wxy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> wyx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> xzw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> xwz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> zxw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> zwx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> wxz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> wzx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> yzw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> ywz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> zyw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> zwy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> wyz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 3> wzy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> xyzw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> xywz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> xzyw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> xzwy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> xwyz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> xwzy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> yxzw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> yxwz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> yzxw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> yzwx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> ywxz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> ywzx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> zxyw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> zxwy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> zyxw() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> zywx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> zwxy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> zwyx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> wxyz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> wxzy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> wyxz() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> wyzx() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> wzxy() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, 4> wzyx() const;

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

		LIBRAPID_ALWAYS_INLINE void xy(const SIMDVector<Scalar, 2> &v);
		LIBRAPID_ALWAYS_INLINE void yx(const SIMDVector<Scalar, 2> &v);
		LIBRAPID_ALWAYS_INLINE void xz(const SIMDVector<Scalar, 2> &v);
		LIBRAPID_ALWAYS_INLINE void zx(const SIMDVector<Scalar, 2> &v);
		LIBRAPID_ALWAYS_INLINE void yz(const SIMDVector<Scalar, 2> &v);
		LIBRAPID_ALWAYS_INLINE void zy(const SIMDVector<Scalar, 2> &v);
		LIBRAPID_ALWAYS_INLINE void xyz(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void xzy(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void yxz(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void yzx(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void zxy(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void zyx(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void xyw(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void xwy(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void yxw(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void ywx(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void wxy(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void wyx(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void xzw(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void xwz(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void zxw(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void zwx(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void wxz(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void wzx(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void yzw(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void ywz(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void zyw(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void zwy(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void wyz(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void wzy(const SIMDVector<Scalar, 3> &v);
		LIBRAPID_ALWAYS_INLINE void xyzw(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void xywz(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void xzyw(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void xzwy(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void xwyz(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void xwzy(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void yxzw(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void yxwz(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void yzxw(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void yzwx(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void ywxz(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void ywzx(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void zxyw(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void zxwy(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void zyxw(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void zywx(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void zwxy(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void zwyx(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void wxyz(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void wxzy(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void wyxz(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void wyzx(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void wzxy(const SIMDVector<Scalar, 4> &v);
		LIBRAPID_ALWAYS_INLINE void wzyx(const SIMDVector<Scalar, 4> &v);

		/// Return the underlying storage type
		/// \return The underlying storage type
		LIBRAPID_ALWAYS_INLINE const StorageType &data() const;

		/// Return the underlying storage type
		/// \return The underlying storage type
		LIBRAPID_ALWAYS_INLINE StorageType &data();

		/// Convert a vector into a string representation -- "(x, y, z, w, ...)"
		/// \param formatString The format string to use for each component
		/// \return A string representation of this vector
		LIBRAPID_NODISCARD std::string str(const std::string &formatString = "{}") const;

	protected:
		StorageType m_data {};
	};

	template<typename Scalar, int64_t Dims>
	SIMDVector<Scalar, Dims>::SIMDVector(const StorageType &arr) : m_data {arr} {}

	template<typename Scalar, int64_t Dims>
	template<typename T, typename ABI>
	SIMDVector<Scalar, Dims>::SIMDVector(const Vc::Vector<T, ABI> &arr) : m_data {arr} {}

	template<typename Scalar, int64_t Dims>
	template<typename S, int64_t D>
	SIMDVector<Scalar, Dims>::SIMDVector(const SIMDVector<S, D> &other) {
		for (int64_t i = 0; i < min(Dims, D); ++i) { m_data[i] = static_cast<Scalar>(other[i]); }
	}

	template<typename Scalar, int64_t Dims>
	template<typename... Args, std::enable_if_t<sizeof...(Args) == Dims, int>>
	SIMDVector<Scalar, Dims>::SIMDVector(Args... args) : m_data {static_cast<Scalar>(args)...} {
		static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
	}

	template<typename Scalar, int64_t Dims>
	template<typename... Args, int64_t size, std::enable_if_t<size != Dims, int>>
	SIMDVector<Scalar, Dims>::SIMDVector(Args... args) {
		static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
		if constexpr (size == 1) {
			m_data = StorageType(static_cast<Scalar>(args)...);
		} else {
			const Scalar expanded[] = {static_cast<Scalar>(args)...};
			for (int64_t i = 0; i < size; i++) { m_data[i] = expanded[i]; }
		}
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	SIMDVector<Scalar, Dims>::SIMDVector(const std::initializer_list<T> &list) {
		assert(list.size() <= Dims);
		int64_t i = 0;
		for (const Scalar &val : list) { m_data[i++] = val; }
	}

	template<typename Scalar, int64_t Dims1, int64_t Dims2>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims1>
	operator&(const SIMDVector<Scalar, Dims1> &vec, const SIMDVector<Scalar, Dims2> &mask) {
		constexpr int64_t m = Dims1 < Dims2 ? Dims1 : Dims2;
		SIMDVector<Scalar, m> res(vec);
		for (size_t i = 0; i < m; ++i)
			if (!mask[i]) res[i] = 0;
		return res;
	}

#ifdef GLM_VERSION

	/// Construct a Vector from a GLM Vector
	template<typename Scalar, int64_t Dims>
	template<glm::qualifier p>
	SIMDVector<Scalar, Dims>::SIMDVector(const glm::vec<Dimscalar, p> &vec) {
		for (size_t i = 0; i < Dims; ++i) { m_data[i] = vec[i]; }
	}

	/// Convert a GLM vector to a Vector object
	template<typename Scalar, int64_t Dims>
	template<glm::qualifier p>
	SIMDVector<Scalar, Dims>::operator glm::vec<Dimscalar, p>() const {
		glm::vec<Dimscalar, p> res;
		for (size_t i = 0; i < Dims; ++i) { res[i] = m_data[i]; }
		return res;
	}

#endif // GLM_VERSION

	template<typename Scalar, int64_t Dims>
	const auto SIMDVector<Scalar, Dims>::operator[](int64_t index) const {
		LIBRAPID_ASSERT(0 <= index && index < Dims,
						"Index {} out of range for vector with {} dimensions",
						index,
						Dims);
		return m_data[index];
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::operator[](int64_t index) {
		LIBRAPID_ASSERT(0 <= index && index < Dims,
						"Index {} out of range for vector with {} dimensions",
						index,
						Dims);
		return m_data[index];
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator+=(const SIMDVector<T, d> &other) -> SIMDVector & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data += other.m_data;
		} else {
			constexpr int64_t m = d < Dims ? d : Dims;
			for (int64_t i = 0; i < m; ++i) { m_data[i] += other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator-=(const SIMDVector<T, d> &other) -> SIMDVector & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data -= other.m_data;
		} else {
			constexpr int64_t m = d < Dims ? d : Dims;
			for (int64_t i = 0; i < m; ++i) { m_data[i] -= other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator*=(const SIMDVector<T, d> &other) -> SIMDVector & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data *= other.m_data;
		} else {
			constexpr int64_t m = d < Dims ? d : Dims;
			for (int64_t i = 0; i < m; ++i) { m_data[i] *= other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator/=(const SIMDVector<T, d> &other) -> SIMDVector & {
		static_assert(d <= Dims, "Invalid number of dimensions");
		if constexpr (Dims == d) {
			m_data /= other.m_data;
		} else {
			constexpr int64_t m = d < Dims ? d : Dims;
			for (int64_t i = 0; i < m; ++i) { m_data[i] /= other[i]; }
		}
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator+=(const T &value) -> SIMDVector & {
		m_data += static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator-=(const T &value) -> SIMDVector & {
		m_data -= static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator*=(const T &value) -> SIMDVector & {
		m_data *= static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator/=(const T &value) -> SIMDVector & {
		m_data /= static_cast<Scalar>(value);
		return *this;
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::operator-() const -> SIMDVector {
		SIMDVector res(*this);
		res *= -1;
		return res;
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::cmp(const SIMDVector<T, d> &other, const char *mode) const
	  -> SIMDVector {
		SIMDVector res(*this);
		int16_t modeInt = *(int16_t *)mode;
		for (size_t i = 0; i < Dims; ++i) {
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

	template<typename Scalar, int64_t Dims>
	template<typename T>
	auto SIMDVector<Scalar, Dims>::cmp(const T &value, const char *mode) const -> SIMDVector {
		// Mode:
		// 0: ==    1: !=
		// 2: <     3: <=
		// 4: >     5: >=

		SIMDVector res(*this);
		int16_t modeInt = *(int16_t *)mode;
		for (size_t i = 0; i < Dims; ++i) {
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

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator<(const SIMDVector<T, d> &other) const -> SIMDVector {
		return cmp(other, "lt");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator<=(const SIMDVector<T, d> &other) const -> SIMDVector {
		return cmp(other, "le");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator>(const SIMDVector<T, d> &other) const -> SIMDVector {
		return cmp(other, "gt");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator>=(const SIMDVector<T, d> &other) const -> SIMDVector {
		return cmp(other, "ge");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator==(const SIMDVector<T, d> &other) const -> SIMDVector {
		return cmp(other, "eq");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, int64_t d>
	auto SIMDVector<Scalar, Dims>::operator!=(const SIMDVector<T, d> &other) const -> SIMDVector {
		return cmp(other, "ne");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator<(const T &other) const -> SIMDVector {
		return cmp(other, "lt");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator<=(const T &other) const -> SIMDVector {
		return cmp(other, "le");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator>(const T &other) const -> SIMDVector {
		return cmp(other, "gt");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator>=(const T &other) const -> SIMDVector {
		return cmp(other, "ge");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator==(const T &other) const -> SIMDVector {
		return cmp(other, "eq");
	}

	template<typename Scalar, int64_t Dims>
	template<typename T, std::enable_if_t<std::is_convertible_v<T, Scalar>, int>>
	auto SIMDVector<Scalar, Dims>::operator!=(const T &other) const -> SIMDVector {
		return cmp(other, "ne");
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::mag2() const -> Scalar {
		return (m_data * m_data).sum();
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::mag() const -> Scalar {
		return sqrt(mag2());
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::norm() const -> SIMDVector {
		SIMDVector res(*this);
		res /= mag();
		return res;
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::dot(const SIMDVector &other) const -> Scalar {
		return (m_data * other.m_data).sum();
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::cross(const SIMDVector &other) const -> SIMDVector {
		static_assert(Dims == 3, "Cross product is only defined for 3D Vectors");
		return SIMDVector(y() * other.z() - z() * other.y(),
						  z() * other.x() - x() * other.z(),
						  x() * other.y() - y() * other.x());
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::proj(const SIMDVector &other) const -> SIMDVector {
		return other * (other.dot(*this) / other.mag2());
	}

	template<typename Scalar, int64_t Dims>
	SIMDVector<Scalar, Dims>::operator bool() const {
		for (size_t i = 0; i < Dims; ++i)
			if (m_data[i] != 0) return true;
		return false;
	}

	/// Add two Vector objects together and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator+(const SIMDVector<Scalar, Dims> &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res += rhs;
		return res;
	}

	/// Subtract two Vector objects and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the subtraction
	/// \param rhs The right hand side of the subtraction
	/// \return The result of the subtraction
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator-(const SIMDVector<Scalar, Dims> &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res -= rhs;
		return res;
	}

	/// Multiply two Vector objects element-by-element and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the multiplication
	/// \param rhs The right hand side of the multiplication
	/// \return The result of the multiplication
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator*(const SIMDVector<Scalar, Dims> &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res *= rhs;
		return res;
	}

	/// Divide two Vector objects element-by-element and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the division
	/// \param rhs The right hand side of the division
	/// \return The result of the division
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator/(const SIMDVector<Scalar, Dims> &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res /= rhs;
		return res;
	}

	/// Add a scalar to a Vector object and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename Scalar, int64_t Dims, typename S>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator+(const SIMDVector<Scalar, Dims> &lhs,
															  const S &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res += rhs;
		return res;
	}

	/// Subtract a scalar from a Vector object and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the subtraction
	/// \param rhs The right hand side of the subtraction
	/// \return The result of the subtraction
	template<typename Scalar, int64_t Dims, typename S>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator-(const SIMDVector<Scalar, Dims> &lhs,
															  const S &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res -= rhs;
		return res;
	}

	/// Multiply a Vector object by a scalar and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the multiplication
	/// \param rhs The right hand side of the multiplication
	/// \return The result of the multiplication
	template<typename Scalar, int64_t Dims, typename S>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator*(const SIMDVector<Scalar, Dims> &lhs,
															  const S &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res *= rhs;
		return res;
	}

	/// Divide a Vector object by a scalar and return the result
	/// \tparam Scalar The type of the scalar
	/// \tparam Dims The number of dimensions
	/// \tparam S The type of the scalar
	/// \param lhs The left hand side of the division
	/// \param rhs The right hand side of the division
	/// \return The result of the division
	template<typename Scalar, int64_t Dims, typename S>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator/(const SIMDVector<Scalar, Dims> &lhs,
															  const S &rhs) {
		SIMDVector<Scalar, Dims> res(lhs);
		res /= rhs;
		return res;
	}

	/// Add a scalar to a Vector object and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the addition
	/// \param rhs The right hand side of the addition
	/// \return The result of the addition
	template<typename S, typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator+(const S &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		using StorageType = typename SIMDVector<Scalar, Dims>::StorageType;
		SIMDVector<Scalar, Dims> res(StorageType(static_cast<Scalar>(lhs)));
		res += rhs;
		return res;
	}

	/// Subtract a Vector object from a scalar and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the subtraction
	/// \param rhs The right hand side of the subtraction
	/// \return The result of the subtraction
	template<typename S, typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator-(const S &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		using StorageType = typename SIMDVector<Scalar, Dims>::StorageType;
		SIMDVector<Scalar, Dims> res(StorageType(static_cast<Scalar>(lhs)));
		res -= rhs;
		return res;
	}

	/// Multiply a scalar by a Vector object and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the multiplication
	/// \param rhs The right hand side of the multiplication
	/// \return The result of the multiplication
	template<typename S, typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator*(const S &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		using StorageType = typename SIMDVector<Scalar, Dims>::StorageType;
		SIMDVector<Scalar, Dims> res(StorageType(static_cast<Scalar>(lhs)));
		res *= rhs;
		return res;
	}

	/// Divide a scalar by a Vector object and return the result
	/// \tparam S The type of the scalar
	/// \tparam Scalar The scalar type of the Vector
	/// \tparam Dims The number of dimensions
	/// \param lhs The left hand side of the division
	/// \param rhs The right hand side of the division
	/// \return The result of the division
	template<typename S, typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> operator/(const S &lhs,
															  const SIMDVector<Scalar, Dims> &rhs) {
		using StorageType = typename SIMDVector<Scalar, Dims>::StorageType;
		SIMDVector<Scalar, Dims> res(StorageType(static_cast<Scalar>(lhs)));
		res /= rhs;
		return res;
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::x() const -> Scalar {
		if constexpr (Dims < 1)
			return 0;
		else
			return m_data[0];
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::y() const -> Scalar {
		if constexpr (Dims < 2)
			return 0;
		else
			return m_data[1];
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::z() const -> Scalar {
		if constexpr (Dims < 3)
			return 0;
		else
			return m_data[2];
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::w() const -> Scalar {
		if constexpr (Dims < 4)
			return 0;
		else
			return m_data[3];
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xy() const -> SIMDVector<Scalar, 2> {
		return {x(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yx() const -> SIMDVector<Scalar, 2> {
		return {y(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xz() const -> SIMDVector<Scalar, 2> {
		return {x(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zx() const -> SIMDVector<Scalar, 2> {
		return {z(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yz() const -> SIMDVector<Scalar, 2> {
		return {y(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zy() const -> SIMDVector<Scalar, 2> {
		return {z(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xyz() const -> SIMDVector<Scalar, 3> {
		return {x(), y(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xzy() const -> SIMDVector<Scalar, 3> {
		return {x(), z(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yxz() const -> SIMDVector<Scalar, 3> {
		return {y(), x(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yzx() const -> SIMDVector<Scalar, 3> {
		return {y(), z(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zxy() const -> SIMDVector<Scalar, 3> {
		return {z(), x(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zyx() const -> SIMDVector<Scalar, 3> {
		return {z(), y(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xyw() const -> SIMDVector<Scalar, 3> {
		return {x(), y(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xwy() const -> SIMDVector<Scalar, 3> {
		return {x(), w(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yxw() const -> SIMDVector<Scalar, 3> {
		return {y(), x(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::ywx() const -> SIMDVector<Scalar, 3> {
		return {y(), w(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wxy() const -> SIMDVector<Scalar, 3> {
		return {w(), x(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wyx() const -> SIMDVector<Scalar, 3> {
		return {w(), y(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xzw() const -> SIMDVector<Scalar, 3> {
		return {x(), z(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xwz() const -> SIMDVector<Scalar, 3> {
		return {x(), w(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zxw() const -> SIMDVector<Scalar, 3> {
		return {z(), x(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zwx() const -> SIMDVector<Scalar, 3> {
		return {z(), w(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wxz() const -> SIMDVector<Scalar, 3> {
		return {w(), x(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wzx() const -> SIMDVector<Scalar, 3> {
		return {w(), z(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yzw() const -> SIMDVector<Scalar, 3> {
		return {y(), z(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::ywz() const -> SIMDVector<Scalar, 3> {
		return {y(), w(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zyw() const -> SIMDVector<Scalar, 3> {
		return {z(), y(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zwy() const -> SIMDVector<Scalar, 3> {
		return {z(), w(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wyz() const -> SIMDVector<Scalar, 3> {
		return {w(), y(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wzy() const -> SIMDVector<Scalar, 3> {
		return {w(), z(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xyzw() const -> SIMDVector<Scalar, 4> {
		return {x(), y(), z(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xywz() const -> SIMDVector<Scalar, 4> {
		return {x(), y(), w(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xzyw() const -> SIMDVector<Scalar, 4> {
		return {x(), z(), y(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xzwy() const -> SIMDVector<Scalar, 4> {
		return {x(), z(), w(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xwyz() const -> SIMDVector<Scalar, 4> {
		return {x(), w(), y(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::xwzy() const -> SIMDVector<Scalar, 4> {
		return {x(), w(), z(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yxzw() const -> SIMDVector<Scalar, 4> {
		return {y(), x(), z(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yxwz() const -> SIMDVector<Scalar, 4> {
		return {y(), x(), w(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yzxw() const -> SIMDVector<Scalar, 4> {
		return {y(), z(), x(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::yzwx() const -> SIMDVector<Scalar, 4> {
		return {y(), z(), w(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::ywxz() const -> SIMDVector<Scalar, 4> {
		return {y(), w(), x(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::ywzx() const -> SIMDVector<Scalar, 4> {
		return {y(), w(), z(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zxyw() const -> SIMDVector<Scalar, 4> {
		return {z(), x(), y(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zxwy() const -> SIMDVector<Scalar, 4> {
		return {z(), x(), w(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zyxw() const -> SIMDVector<Scalar, 4> {
		return {z(), y(), x(), w()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zywx() const -> SIMDVector<Scalar, 4> {
		return {z(), y(), w(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zwxy() const -> SIMDVector<Scalar, 4> {
		return {z(), w(), x(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::zwyx() const -> SIMDVector<Scalar, 4> {
		return {z(), w(), y(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wxyz() const -> SIMDVector<Scalar, 4> {
		return {w(), x(), y(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wxzy() const -> SIMDVector<Scalar, 4> {
		return {w(), x(), z(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wyxz() const -> SIMDVector<Scalar, 4> {
		return {w(), y(), x(), z()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wyzx() const -> SIMDVector<Scalar, 4> {
		return {w(), y(), z(), x()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wzxy() const -> SIMDVector<Scalar, 4> {
		return {w(), z(), x(), y()};
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::wzyx() const -> SIMDVector<Scalar, 4> {
		return {w(), z(), y(), x()};
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::x(Scalar val) {
		if constexpr (Dims >= 1) m_data[0] = val;
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::y(Scalar val) {
		if constexpr (Dims >= 2) m_data[1] = val;
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::z(Scalar val) {
		if constexpr (Dims >= 3) m_data[2] = val;
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::w(Scalar val) {
		if constexpr (Dims >= 4) m_data[3] = val;
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xy(const SIMDVector<Scalar, 2> &v) {
		x(v.x());
		y(v.y());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yx(const SIMDVector<Scalar, 2> &v) {
		y(v.x());
		x(v.y());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xz(const SIMDVector<Scalar, 2> &v) {
		x(v.x());
		z(v.y());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zx(const SIMDVector<Scalar, 2> &v) {
		z(v.x());
		x(v.y());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yz(const SIMDVector<Scalar, 2> &v) {
		y(v.x());
		z(v.y());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zy(const SIMDVector<Scalar, 2> &v) {
		z(v.x());
		y(v.y());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xyz(const SIMDVector<Scalar, 3> &v) {
		x(v.x());
		y(v.y());
		z(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xzy(const SIMDVector<Scalar, 3> &v) {
		x(v.x());
		z(v.y());
		y(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yxz(const SIMDVector<Scalar, 3> &v) {
		y(v.x());
		x(v.y());
		z(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yzx(const SIMDVector<Scalar, 3> &v) {
		y(v.x());
		z(v.y());
		x(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zxy(const SIMDVector<Scalar, 3> &v) {
		z(v.x());
		x(v.y());
		y(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zyx(const SIMDVector<Scalar, 3> &v) {
		z(v.x());
		y(v.y());
		x(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xyw(const SIMDVector<Scalar, 3> &v) {
		x(v.x());
		y(v.y());
		w(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xwy(const SIMDVector<Scalar, 3> &v) {
		x(v.x());
		w(v.y());
		y(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yxw(const SIMDVector<Scalar, 3> &v) {
		y(v.x());
		x(v.y());
		w(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::ywx(const SIMDVector<Scalar, 3> &v) {
		y(v.x());
		w(v.y());
		x(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wxy(const SIMDVector<Scalar, 3> &v) {
		w(v.x());
		x(v.y());
		y(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wyx(const SIMDVector<Scalar, 3> &v) {
		w(v.x());
		y(v.y());
		x(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xzw(const SIMDVector<Scalar, 3> &v) {
		x(v.x());
		z(v.y());
		w(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xwz(const SIMDVector<Scalar, 3> &v) {
		x(v.x());
		w(v.y());
		z(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zxw(const SIMDVector<Scalar, 3> &v) {
		z(v.x());
		x(v.y());
		w(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zwx(const SIMDVector<Scalar, 3> &v) {
		z(v.x());
		w(v.y());
		x(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wxz(const SIMDVector<Scalar, 3> &v) {
		w(v.x());
		x(v.y());
		z(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wzx(const SIMDVector<Scalar, 3> &v) {
		w(v.x());
		z(v.y());
		x(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yzw(const SIMDVector<Scalar, 3> &v) {
		y(v.x());
		z(v.y());
		w(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::ywz(const SIMDVector<Scalar, 3> &v) {
		y(v.x());
		w(v.y());
		z(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zyw(const SIMDVector<Scalar, 3> &v) {
		z(v.x());
		y(v.y());
		w(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zwy(const SIMDVector<Scalar, 3> &v) {
		z(v.x());
		w(v.y());
		y(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wyz(const SIMDVector<Scalar, 3> &v) {
		w(v.x());
		y(v.y());
		z(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wzy(const SIMDVector<Scalar, 3> &v) {
		w(v.x());
		z(v.y());
		y(v.z());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xyzw(const SIMDVector<Scalar, 4> &v) {
		x(v.x());
		y(v.y());
		z(v.z());
		w(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xywz(const SIMDVector<Scalar, 4> &v) {
		x(v.x());
		y(v.y());
		w(v.z());
		z(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xzyw(const SIMDVector<Scalar, 4> &v) {
		x(v.x());
		z(v.y());
		y(v.z());
		w(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xzwy(const SIMDVector<Scalar, 4> &v) {
		x(v.x());
		z(v.y());
		w(v.z());
		y(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xwyz(const SIMDVector<Scalar, 4> &v) {
		x(v.x());
		w(v.y());
		y(v.z());
		z(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::xwzy(const SIMDVector<Scalar, 4> &v) {
		x(v.x());
		w(v.y());
		z(v.z());
		y(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yxzw(const SIMDVector<Scalar, 4> &v) {
		y(v.x());
		x(v.y());
		z(v.z());
		w(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yxwz(const SIMDVector<Scalar, 4> &v) {
		y(v.x());
		x(v.y());
		w(v.z());
		z(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yzxw(const SIMDVector<Scalar, 4> &v) {
		y(v.x());
		z(v.y());
		x(v.z());
		w(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::yzwx(const SIMDVector<Scalar, 4> &v) {
		y(v.x());
		z(v.y());
		w(v.z());
		x(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::ywxz(const SIMDVector<Scalar, 4> &v) {
		y(v.x());
		w(v.y());
		x(v.z());
		z(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::ywzx(const SIMDVector<Scalar, 4> &v) {
		y(v.x());
		w(v.y());
		z(v.z());
		x(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zxyw(const SIMDVector<Scalar, 4> &v) {
		z(v.x());
		x(v.y());
		y(v.z());
		w(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zxwy(const SIMDVector<Scalar, 4> &v) {
		z(v.x());
		x(v.y());
		w(v.z());
		y(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zyxw(const SIMDVector<Scalar, 4> &v) {
		z(v.x());
		y(v.y());
		x(v.z());
		w(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zywx(const SIMDVector<Scalar, 4> &v) {
		z(v.x());
		y(v.y());
		w(v.z());
		x(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zwxy(const SIMDVector<Scalar, 4> &v) {
		z(v.x());
		w(v.y());
		x(v.z());
		y(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::zwyx(const SIMDVector<Scalar, 4> &v) {
		z(v.x());
		w(v.y());
		y(v.z());
		x(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wxyz(const SIMDVector<Scalar, 4> &v) {
		w(v.x());
		x(v.y());
		y(v.z());
		z(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wxzy(const SIMDVector<Scalar, 4> &v) {
		w(v.x());
		x(v.y());
		z(v.z());
		y(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wyxz(const SIMDVector<Scalar, 4> &v) {
		w(v.x());
		y(v.y());
		x(v.z());
		z(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wyzx(const SIMDVector<Scalar, 4> &v) {
		w(v.x());
		y(v.y());
		z(v.z());
		x(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wzxy(const SIMDVector<Scalar, 4> &v) {
		w(v.x());
		z(v.y());
		x(v.z());
		y(v.w());
	}

	template<typename Scalar, int64_t Dims>
	void SIMDVector<Scalar, Dims>::wzyx(const SIMDVector<Scalar, 4> &v) {
		w(v.x());
		z(v.y());
		y(v.z());
		x(v.w());
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::data() const -> const StorageType & {
		return m_data;
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::data() -> StorageType & {
		return m_data;
	}

	template<typename Scalar, int64_t Dims>
	auto SIMDVector<Scalar, Dims>::str(const std::string &formatString) const -> std::string {
		std::string res = "(";
		for (size_t i = 0; i < Dims; ++i) {
			res += fmt::format(formatString, m_data[i]);
			if (i < Dims - 1) res += ", ";
		}
		return res + ")";
	}

	/// Calculate the squared distance between two vectors
	/// \tparam Scalar The scalar type of the vectors
	/// \tparam Dims The dimensionality of the vectors
	/// \param lhs The first vector
	/// \param rhs The second vector
	/// \return The squared distance between the two vectors
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE Scalar dist2(const SIMDVector<Scalar, Dims> &lhs,
										const SIMDVector<Scalar, Dims> &rhs) {
		return (lhs - rhs).mag2();
	}

	/// Calculate the distance between two vectors
	/// \tparam Scalar The scalar type of the vectors
	/// \tparam Dims The dimensionality of the vectors
	/// \param lhs The first vector
	/// \param rhs The second vector
	/// \return The distance between the two vectors
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE Scalar dist(const SIMDVector<Scalar, Dims> &lhs,
									   const SIMDVector<Scalar, Dims> &rhs) {
		return (lhs - rhs).mag();
	}

	/// Calculate the sin of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the sin of
	/// \return The sin of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> sin(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return SIMDVector<Scalar, Dims>(Vc::sin(vec.data()));
	}

	/// Calculate the cos of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the cos of
	/// \return The cos of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> cos(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return SIMDVector<Scalar, Dims>(Vc::cos(vec.data()));
	}

	/// Calculate the tan of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the tan of
	/// \return The tan of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> tan(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return sin(vec) / cos(vec);
	}

	/// Calculate the asin of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the asin of
	/// \return The asin of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> asin(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return SIMDVector<Scalar, Dims>(Vc::asin(vec.data()));
	}

	/// Calculate the acos of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the acos of
	/// \return The acos of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> acos(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		// Vc doesn't have acos
		return Scalar(HALFPI) - asin(vec);
	}

	/// Calculate the atan of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the atan of
	/// \return The atan of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> atan(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return SIMDVector<Scalar, Dims>(Vc::atan(vec.data()));
	}

	/// Calculate the atan2 of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the atan2 of
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> atan2(const SIMDVector<Scalar, Dims> &lhs,
														  const SIMDVector<Scalar, Dims> &rhs) {
		using Type = SIMDVector<Scalar, Dims>;
		return SIMDVector<Scalar, Dims>(Vc::atan2(lhs.data(), rhs.data()));
	}

	/// Calculate the sinh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the sinh of
	/// \return The sinh of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> sinh(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::sinh(static_cast<Scalar>(vec[i]));
		}
		return res;
	}

	/// Calculate the cosh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the cosh of
	/// \return The cosh of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> cosh(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::cosh(static_cast<Scalar>(vec[i]));
		}
		return res;
	}

	/// Calculate the tanh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the tanh of
	/// \return The tanh of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> tanh(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::tanh(static_cast<Scalar>(vec[i]));
		}
		return res;
	}

	/// Calculate the asinh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the asinh of
	/// \return The asinh of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> asinh(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::asinh(static_cast<Scalar>(vec[i]));
		}
		return res;
	}

	/// Calculate the acosh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the acosh of
	/// \return The acosh of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> acosh(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::acosh(static_cast<Scalar>(vec[i]));
		}
		return res;
	}

	/// Calculate the atanh of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the atanh of
	/// \return The atanh of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> atanh(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::atanh(static_cast<Scalar>(vec[i]));
		}
		return res;
	}

	/// Calculate the exp of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the exp of
	/// \return The exp of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> exp(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::exp(vec.data()));
	}

	/// Calculate the log of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the log of
	/// \return The log of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> log(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::log(vec.data()));
	}

	/// Calculate the log10 of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the log10 of
	/// \return The log10 of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> log10(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::log10(vec.data()));
	}

	/// Calculate the log2 of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the log2 of
	/// \return The log2 of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> log2(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::log2(vec.data()));
	}

	/// Calculate the sqrt of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the sqrt of
	/// \return The sqrt of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> sqrt(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::sqrt(vec.data()));
	}

	/// Raise each element of a vector to the power of another vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec Base vector
	/// \param exp Vector of exponents
	/// \return The result of raising each element of the vector to the power of the corresponding
	///         element of the exponent vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> pow(const SIMDVector<Scalar, Dims> &vec,
														const SIMDVector<Scalar, Dims> &exp) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] = ::librapid::pow(static_cast<Scalar>(vec[i]), static_cast<Scalar>(exp[i]));
		}
		return res;
	}

	/// Raise each element of a vector to the power of a scalar and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \tparam T The scalar type of the exponent
	/// \param vec Base vector
	/// \param exp Scalar exponent
	/// \return The result of raising each element of the vector to the power of the scalar
	template<typename Scalar, int64_t Dims, typename T>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> pow(const SIMDVector<Scalar, Dims> &vec,
														T exp) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) {
			res[i] =
			  static_cast<Scalar>(pow(static_cast<Scalar>(vec[i]), static_cast<Scalar>(exp)));
		}
		return res;
	}

	/// Raise a scalar to the power of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec Base vector
	/// \param exp Scalar exponent
	/// \return The result of raising the scalar to the power of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> pow(Scalar vec,
														const SIMDVector<Scalar, Dims> &exp) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) { res[i] = ::librapid::pow(vec, exp[i]); }
		return res;
	}

	/// Calculate the cbrt of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the cbrt of
	/// \return The cbrt of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> cbrt(const SIMDVector<Scalar, Dims> &vec) {
		SIMDVector<Scalar, Dims> res;
		for (size_t i = 0; i < Dims; ++i) { res[i] = ::librapid::cbrt(vec[i]); }
		return res;
	}

	/// Calculate the absolute value of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the absolute value of
	/// \return The absolute value of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> abs(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::abs(vec.data()));
	}

	/// Calculate the floor of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the floor of
	/// \return The floor of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> floor(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::floor(vec.data()));
	}

	/// Calculate the ceil of each element of a vector and return the result
	/// \tparam Scalar The scalar type of the vector
	/// \tparam Dims The dimensionality of the vector
	/// \param vec The vector to calculate the ceil of
	/// \return The ceil of each element of the vector
	template<typename Scalar, int64_t Dims>
	LIBRAPID_ALWAYS_INLINE SIMDVector<Scalar, Dims> ceil(const SIMDVector<Scalar, Dims> &vec) {
		using Type = SIMDVector<Scalar, Dims>;
		return Type(Vc::ceil(vec.data()));
	}

	/// \brief Returns true if the two vectors are within the given tolerance of each other
	///
	/// \tparam Scalar The scalar type of the vectors
	/// \tparam Dims Number of dimensions of the vectors
	/// \param a The first vector
	/// \param b The second vector
	/// \param tolerance Tolerance
	/// \return True if the vectors are within the given tolerance of each other
	/// \see isClose(const GenericVector<Scalar, Dims> &, const GenericVector<Scalar, Dims> &)
	template<typename Scalar, int64_t Dims>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isClose(const SIMDVector<Scalar, Dims> &a,
														   const SIMDVector<Scalar, Dims> &b,
														   Scalar tolerance = -1) {
		if (tolerance < 0) {
			if constexpr (std::is_same_v<Scalar, double>) {
				tolerance = 1e-12;
			} else if constexpr (std::is_same_v<Scalar, float>) {
				tolerance = 1e-6f;
			} else if constexpr (std::is_floating_point_v<Scalar>) {
				tolerance = 1e-4;
			} else {
				tolerance = 0;
			}
		};

		return (a - b).mag2() <= tolerance;
	}

	template<typename Scalar, int64_t Dims>
	std::ostream &operator<<(std::ostream &os, const SIMDVector<Scalar, Dims> &vec) {
		os << vec.str();
		return os;
	}
} // namespace librapid

#ifdef FMT_API
template<typename Scalar, int64_t D>
struct fmt::formatter<librapid::SIMDVector<Scalar, D>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::SIMDVector<Scalar, D> &arr, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), arr.str());
	}
};
#endif // FMT_API

#endif // LIBRAPID_MATH_VECTOR_HPP
