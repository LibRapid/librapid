#ifndef LIBRAPID_VECTOR
#define LIBRAPID_VECTOR

namespace librapid {
#ifndef LIBRAPID_DOXYGEN_BUILD
#	define MIN_DIM_CLAMP(_dims, _tmpDims) (((_dims) < (_tmpDims)) ? (_dims) : (_tmpDims))
#else
#	define MIN_DIM_CLAMP(_dims, _tmpDims) _dims
#endif

#ifndef LIBRAPID_DOXYGEN_BUILD
#	define MAX_DIM_CLAMP(_dims, _tmpDims) (((_dims) > (_tmpDims)) ? (_dims) : (_tmpDims))
#else
#	define MAX_DIM_CLAMP(_dims, _tmpDims) _dims
#endif

	/*
	template<typename DTYPE, i64 dims>
	class Vec {
		template<typename T>
		using Common = typename std::common_type<DTYPE, T>::type;

	public:
		Vec() = default;

		template<typename X, typename... YZ>
		explicit Vec(X x, YZ... yz) : m_data {(DTYPE)x, (DTYPE)yz...} {
			static_assert(sizeof...(YZ) <= dims, "Parameters cannot exceed vector dimensions");
		}

		template<typename T>
		Vec(const std::initializer_list<T> &vals) {
			LR_ASSERT(vals.size() <= dims, "Parameters cannot exceed vector dimensions");
			i64 ind = 0;
			for (const auto &val : vals) { m_data[ind++] = val; }
		}

		template<typename T, i64 d>
		explicit Vec(const Vec<T, d> &other) {
			i64 i;
			for (i = 0; i < MIN_DIM_CLAMP(dims, d); ++i) { m_data[i] = other[i]; }
		}

		template<typename T>
		Vec(const Vec<T, 3> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec(const Vec<DTYPE, dims> &other) {
			i64 i;
			for (i = 0; i < dims; ++i) { m_data[i] = other[i]; }
		}

		Vec<DTYPE, dims> &operator=(const Vec<DTYPE, dims> &other) {
			if (this == &other) { return *this; }
			for (i64 i = 0; i < dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<tmpDim, T, p> &vec) {
			for (i64 i = 0; i < tmpDim; ++i) { m_data[i] = (i < dims) ? ((T)vec[i]) : (T()); }
		}

		template<typename T = DTYPE, int tmpDim = dims, glm::qualifier p = glm::defaultp>
		operator glm::vec<tmpDim, T, p>() const {
			glm::vec<tmpDim, T, p> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = (i < dims) ? ((T)m_data[i]) : (T()); }
			return res;
		}

#endif // GLM_VERSION

		// Implement indexing (const and non-const)
		// Functions take a single index and return a scalar value

		const DTYPE &operator[](i64 index) const { return m_data[index]; }

		DTYPE &operator[](i64 index) { return m_data[index]; }

		template<typename T, i64 tmpDims>
		bool operator==(const Vec<T, tmpDims> &other) const {
			// For vectors with different dimensions, return true if the excess
			// values are all zero
			for (i64 i = 0; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				if (m_data[i] != other[i]) return false;
			}

			// Quick return to avoid excess checks
			if (dims == tmpDims) return true;

			for (i64 i = MIN_DIM_CLAMP(dims, tmpDims); i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims && m_data[i]) return false;
				if (i < tmpDims && other[i]) return false;
			}

			return true;
		}

		template<typename T, i64 tmpDims>
		bool operator!=(const Vec<T, tmpDims> &other) const {
			return !(*this == other);
		}

		// Implement equality checks with GLM types
#ifdef GLM_VERSION

		template<typename T, int tmpDims, glm::qualifier p = glm::defaultp>
		bool operator==(const glm::vec<tmpDims, T, p> &other) const {
			// For vectors with different dimensions, return true if the excess
			// values are all zero
			for (i64 i = 0; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				if (m_data[i] != other[i]) return false;
			}

			// Quick return to avoid excess checks
			if (dims == tmpDims) return true;

			for (i64 i = MIN_DIM_CLAMP(dims, tmpDims); i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims && m_data[i]) return false;
				if (i < tmpDims && other[i]) return false;
			}

			return true;
		}

		template<typename T, int tmpDims, glm::qualifier p = glm::defaultp>
		bool operator!=(const glm::vec<tmpDims, T, p> &other) const {
			return !(*this == other);
		}

#endif // GLM_VERSION


		// Implement unary operators


		Vec<DTYPE, dims> operator-() const {
			Vec<DTYPE, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = -m_data[i]; }
			return res;
		}

		// Implement simple arithmetic operators + - * /
		//
		// Operations take two Vec objects and return a new vector (with common
		// type) containing the result of the element-wise operation.
		//
		// Vectors must have same dimensions. To cast, use Vec.as<TYPE, DIMS>()
		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_data[i] : 0) + ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_data[i] : 0) - ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_data[i] : 0) * ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_data[i] : 0) / ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		//
		// Implement simple arithmetic operators + - * /
		//
		// Operations take a vector and a scalar, and return a new vector (with
		// common type) containing the result of the element-wise operation.
		//

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator+(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_data[i] + other; }
			return res;
		}

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator-(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_data[i] - other; }
			return res;
		}

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator*(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_data[i] * other; }
			return res;
		}

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator/(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_data[i] / other; }
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator+=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] += (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator-=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] -= (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator*=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] *= (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator/=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] /= (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator+=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] += other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator-=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] -= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator*=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] *= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator/=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_data[i] /= other; }
			return *this;
		}

		//
		// Return the magnitude squared of a vector
		//
		DTYPE mag2() const {
			DTYPE res = 0;
			for (i64 i = 0; i < dims; ++i) { res += m_data[i] * m_data[i]; }
			return res;
		}

		//
		// Return the magnitude of a vector
		//
		DTYPE mag() const { return sqrt(mag2()); }

		DTYPE invMag() const { return DTYPE(1) / sqrt(mag2()); }

		template<typename T, i64 tmpDims>
		typename std::common_type<DTYPE, T>::type dist2(const Vec<T, tmpDims> &other) const {
			using RET	= typename std::common_type<DTYPE, T>::type;
			RET squared = 0;
			i64 i		= 0;

			// Compute the squares of the differences for the matching
			// components
			for (; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				squared += (m_data[i] - other[i]) * (m_data[i] - other[i]);
			}

			// Compute the squares of the values for the remaining values.
			// This just enables calculating the distance between two vectors
			// with different dimensions
			for (; i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims)
					squared += m_data[i] * m_data[i];
				else
					squared += other[i] * other[i];
			}

			return squared;
		}

		template<typename T, i64 tmpDims>
		typename std::common_type<DTYPE, T>::type dist(const Vec<T, tmpDims> &other) const {
			return sqrt(dist2(other));
		}

		//
		// Compute the vector dot product
		// AxBx + AyBy + AzCz + ...
		//
		template<typename T>
		Common<T> dot(const Vec<Scalar, Dims, StorageType> &other) const {
			Common<T> res = 0;
			for (i64 i = 0; i < dims; ++i) { res += m_data[i] * other[i]; }
			return res;
		}

		//
		// Compute the vector cross product
		//
		template<typename T>
		Vec<Common<T>, dims> cross(const Vec<Scalar, Dims, StorageType> &other) const {
			static_assert(dims == 2 || dims == 3,
						  "Only 2D and 3D vectors support the cross product");

			Vec<Common<T>, dims> res;

			if constexpr (dims == 2) {
				m_data[2] = 0;
				other[2]		= 0;
			}

			res.x = y * other.z - z * other.y;
			res.y = z * other.x - x * other.z;
			res.z = x * other.y - y * other.x;

			return res;
		}

		inline Vec<DTYPE, 2> xy() const { return {x, y}; }

		inline Vec<DTYPE, 2> yx() const { return {y, x}; }

		inline Vec<DTYPE, 3> xyz() const { return {x, y, z}; }

		inline Vec<DTYPE, 3> xzy() const { return {x, z, y}; }

		inline Vec<DTYPE, 3> yxz() const { return {y, x, z}; }

		inline Vec<DTYPE, 3> yzx() const { return {y, z, x}; }

		inline Vec<DTYPE, 3> zxy() const { return {z, x, y}; }

		inline Vec<DTYPE, 3> zyx() const { return {z, y, x}; }

		inline Vec<DTYPE, 4> xyzw() const { return {x, y, z, w}; }

		inline Vec<DTYPE, 4> xywz() const { return {x, y, w, z}; }

		inline Vec<DTYPE, 4> xzyw() const { return {x, z, y, w}; }

		inline Vec<DTYPE, 4> xzwy() const { return {x, z, w, y}; }

		inline Vec<DTYPE, 4> xwyz() const { return {x, w, y, z}; }

		inline Vec<DTYPE, 4> xwzy() const { return {x, w, z, y}; }

		inline Vec<DTYPE, 4> yxzw() const { return {y, x, z, w}; }

		inline Vec<DTYPE, 4> yxwz() const { return {y, x, w, z}; }

		inline Vec<DTYPE, 4> yzxw() const { return {y, z, x, w}; }

		inline Vec<DTYPE, 4> yzwx() const { return {y, z, w, x}; }

		inline Vec<DTYPE, 4> ywxz() const { return {y, w, x, z}; }

		inline Vec<DTYPE, 4> ywzx() const { return {y, w, z, x}; }

		inline Vec<DTYPE, 4> zxyw() const { return {z, x, y, w}; }

		inline Vec<DTYPE, 4> zxwy() const { return {z, x, w, y}; }

		inline Vec<DTYPE, 4> zyxw() const { return {z, y, x, w}; }

		inline Vec<DTYPE, 4> zywx() const { return {z, y, w, x}; }

		inline Vec<DTYPE, 4> zwxy() const { return {z, w, x, y}; }

		inline Vec<DTYPE, 4> zwyx() const { return {z, w, y, x}; }

		inline Vec<DTYPE, 4> wxyz() const { return {w, x, y, z}; }

		inline Vec<DTYPE, 4> wxzy() const { return {w, x, z, y}; }

		inline Vec<DTYPE, 4> wyxz() const { return {w, y, x, z}; }

		inline Vec<DTYPE, 4> wyzx() const { return {w, y, z, x}; }

		inline Vec<DTYPE, 4> wzxy() const { return {w, z, x, y}; }

		inline Vec<DTYPE, 4> wzyx() const { return {w, z, y, x}; }

		[[nodiscard]] std::string str() const {
			std::string res = "(";
			for (i64 i = 0; i < dims; ++i) {
				res += ::librapid::str(m_data[i]) + (i == dims - 1 ? ")" : ", ");
			}
			return res;
		}

		void setX(DTYPE val) { x = val; }

		void setY(DTYPE val) { y = val; }

		void setZ(DTYPE val) { z = val; }

		void setW(DTYPE val) { w = val; }

		DTYPE getX() { return x; }

		DTYPE getY() { return y; }

		DTYPE getZ() { return z; }

		DTYPE getW() { return w; }

		DTYPE &x = m_data[0];
		DTYPE &y = m_data[1];
		DTYPE &z = m_data[2];
		DTYPE &w = m_data[3];

	private:
		DTYPE m_data[dims < 4 ? 4 : dims];
	};

	//
	// Implement simple arithmetic operators + - * /
	//
	// Operations take a scalar and a vector and return a new vector (with
	// common type) containing the result of the element-wise operation.
	//

	template<typename T, typename DTYPE, i64 dims,
			 typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator+(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (i64 i = 0; i < dims; ++i) { res[i] = value + vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, i64 dims,
			 typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator-(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (i64 i = 0; i < dims; ++i) { res[i] = value - vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, i64 dims,
			 typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator*(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (i64 i = 0; i < dims; ++i) { res[i] = value * vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, i64 dims,
			 typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator/(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (i64 i = 0; i < dims; ++i) { res[i] = value / vec[i]; }
		return res;
	}

#define IMPL_VEC_OP(OP)                                                                            \
	template<typename DTYPE, i64 dims>                                                             \
	LR_INLINE Vec<DTYPE, dims> OP(const Vec<DTYPE, dims> &vec) {                                   \
		Vec<DTYPE, dims> res;                                                                      \
		for (i64 i = 0; i < dims; ++i) { res[i] = ::librapid::OP(vec[i]); }                        \
		return res;                                                                                \
	}

	IMPL_VEC_OP(sin)
	IMPL_VEC_OP(cos)
	IMPL_VEC_OP(tan)
	IMPL_VEC_OP(asin)
	IMPL_VEC_OP(acos)
	IMPL_VEC_OP(atan)
	IMPL_VEC_OP(sinh)
	IMPL_VEC_OP(cosh)
	IMPL_VEC_OP(tanh)
	IMPL_VEC_OP(asinh)
	IMPL_VEC_OP(acosh)
	IMPL_VEC_OP(atanh)
	IMPL_VEC_OP(exp)
	IMPL_VEC_OP(exp2)
	IMPL_VEC_OP(sqrt)

	using Vec2i = Vec<i64, 2>;
	using Vec2f = Vec<f32, 2>;
	using Vec2d = Vec<f64, 2>;

	using Vec3i = Vec<i64, 3>;
	using Vec3f = Vec<f32, 3>;
	using Vec3d = Vec<f64, 3>;

	using Vec4i = Vec<i64, 4>;
	using Vec4f = Vec<f32, 4>;
	using Vec4d = Vec<f64, 4>;

#if defined(LIBRAPID_USE_MULTIPREC)
	using Vec2mpfr = Vec<mpfr, 2>;
	using Vec3mpfr = Vec<mpfr, 3>;
	using Vec4mpfr = Vec<mpfr, 4>;
#endif

	template<typename Scalar, i64 Dims, typename StorageType>
	std::ostream &operator<<(std::ostream &os, const Vec<Scalar, Dims, StorageType> &vec) {
		return os << vec.str();
	}
	*/

	template<typename Scalar, i64 Dims, typename StorageType>
	class VecImpl {
	public:
		using Mask = Vc::Mask<Scalar, struct Vc::simd_abi::fixed_size<Dims>>;

		VecImpl() = default;

		explicit VecImpl(const StorageType &arr) : m_data {arr} {}

		template<typename... Args>
		VecImpl(Args... args) : m_data {static_cast<Scalar>(args)...} {
			static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
		}

		VecImpl(const VecImpl &other)				 = default;
		VecImpl(VecImpl &&other) noexcept			 = default;
		VecImpl &operator=(const VecImpl &other)	 = default;
		VecImpl &operator=(VecImpl &&other) noexcept = default;

		LR_NODISCARD("") auto operator[](i64 index) const { return m_data[index]; }

		LR_NODISCARD("") auto &operator[](i64 index) { return m_data[index]; }

		LR_FORCE_INLINE void operator+=(const VecImpl &other) { m_data += other.m_data; }
		LR_FORCE_INLINE void operator-=(const VecImpl &other) { m_data -= other.m_data; }
		LR_FORCE_INLINE void operator*=(const VecImpl &other) { m_data *= other.m_data; }
		LR_FORCE_INLINE void operator/=(const VecImpl &other) { m_data /= other.m_data; }

		LR_FORCE_INLINE void operator+=(const Scalar &value) { m_data += value; }
		LR_FORCE_INLINE void operator-=(const Scalar &value) { m_data -= value; }
		LR_FORCE_INLINE void operator*=(const Scalar &value) { m_data *= value; }
		LR_FORCE_INLINE void operator/=(const Scalar &value) { m_data /= value; }

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator>(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data > other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator<(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data < other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator>=(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data >= other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator<=(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data <= other.m_data);
		}

		LR_NODISCARD("") LR_INLINE Scalar mag2() const { return (m_data * m_data).sum(); }
		LR_NODISCARD("") LR_INLINE Scalar mag() const { return ::librapid::sqrt(mag2()); }
		LR_NODISCARD("") LR_INLINE Scalar invMag() const { return 1.0 / mag(); }

		LR_NODISCARD("") LR_INLINE VecImpl norm() const {
			VecImpl res(*this);
			res /= mag();
			return res;
		}

		LR_NODISCARD("") LR_INLINE Scalar dot(const VecImpl &other) const {
			return (m_data * other.m_data).sum();
		}

		LR_NODISCARD("") LR_INLINE VecImpl cross(const VecImpl &other) const {
			static_assert(Dims == 3, "Cross product is only defined for 3D VecImpltors");
			return VecImpl {m_data[1] * other.m_data[2] - m_data[2] * other.m_data[1],
							m_data[2] * other.m_data[0] - m_data[0] * other.m_data[2],
							m_data[0] * other.m_data[1] - m_data[1] * other.m_data[0]};
		}

		inline VecImpl<Scalar, 2, StorageType> xy() const { return {x(), y()}; }
		inline VecImpl<Scalar, 2, StorageType> yx() const { return {y(), x()}; }
		inline VecImpl<Scalar, 3, StorageType> xyz() const { return {x(), y(), z()}; }
		inline VecImpl<Scalar, 3, StorageType> xzy() const { return {x(), z(), y()}; }
		inline VecImpl<Scalar, 3, StorageType> yxz() const { return {y(), x(), z()}; }
		inline VecImpl<Scalar, 3, StorageType> yzx() const { return {y(), z(), x()}; }
		inline VecImpl<Scalar, 3, StorageType> zxy() const { return {z(), x(), y()}; }
		inline VecImpl<Scalar, 3, StorageType> zyx() const { return {z(), y(), x()}; }
		inline VecImpl<Scalar, 4, StorageType> xyzw() const { return {x(), y(), z(), w()}; }
		inline VecImpl<Scalar, 4, StorageType> xywz() const { return {x(), y(), w(), z()}; }
		inline VecImpl<Scalar, 4, StorageType> xzyw() const { return {x(), z(), y(), w()}; }
		inline VecImpl<Scalar, 4, StorageType> xzwy() const { return {x(), z(), w(), y()}; }
		inline VecImpl<Scalar, 4, StorageType> xwyz() const { return {x(), w(), y(), z()}; }
		inline VecImpl<Scalar, 4, StorageType> xwzy() const { return {x(), w(), z(), y()}; }
		inline VecImpl<Scalar, 4, StorageType> yxzw() const { return {y(), x(), z(), w()}; }
		inline VecImpl<Scalar, 4, StorageType> yxwz() const { return {y(), x(), w(), z()}; }
		inline VecImpl<Scalar, 4, StorageType> yzxw() const { return {y(), z(), x(), w()}; }
		inline VecImpl<Scalar, 4, StorageType> yzwx() const { return {y(), z(), w(), x()}; }
		inline VecImpl<Scalar, 4, StorageType> ywxz() const { return {y(), w(), x(), z()}; }
		inline VecImpl<Scalar, 4, StorageType> ywzx() const { return {y(), w(), z(), x()}; }
		inline VecImpl<Scalar, 4, StorageType> zxyw() const { return {z(), x(), y(), w()}; }
		inline VecImpl<Scalar, 4, StorageType> zxwy() const { return {z(), x(), w(), y()}; }
		inline VecImpl<Scalar, 4, StorageType> zyxw() const { return {z(), y(), x(), w()}; }
		inline VecImpl<Scalar, 4, StorageType> zywx() const { return {z(), y(), w(), x()}; }
		inline VecImpl<Scalar, 4, StorageType> zwxy() const { return {z(), w(), x(), y()}; }
		inline VecImpl<Scalar, 4, StorageType> zwyx() const { return {z(), w(), y(), x()}; }
		inline VecImpl<Scalar, 4, StorageType> wxyz() const { return {w(), x(), y(), z()}; }
		inline VecImpl<Scalar, 4, StorageType> wxzy() const { return {w(), x(), z(), y()}; }
		inline VecImpl<Scalar, 4, StorageType> wyxz() const { return {w(), y(), x(), z()}; }
		inline VecImpl<Scalar, 4, StorageType> wyzx() const { return {w(), y(), z(), x()}; }
		inline VecImpl<Scalar, 4, StorageType> wzxy() const { return {w(), z(), x(), y()}; }
		inline VecImpl<Scalar, 4, StorageType> wzyx() const { return {w(), z(), y(), x()}; }

		LR_FORCE_INLINE Scalar x() const {
			if constexpr (Dims < 1)
				return 0;
			else
				return m_data[0];
		}

		LR_FORCE_INLINE Scalar y() const {
			if constexpr (Dims < 2)
				return 0;
			else
				return m_data[1];
		}

		LR_FORCE_INLINE Scalar z() const {
			if constexpr (Dims < 3)
				return 0;
			else
				return m_data[2];
		}

		LR_FORCE_INLINE Scalar w() const {
			if constexpr (Dims < 4)
				return 0;
			else
				return m_data[3];
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "(";
			for (i64 i = 0; i < Dims; ++i) {
				res += std::to_string(m_data[i]);
				if (i != Dims - 1) { res += ", "; }
			}
			return res + ")";
		}

	protected:
		StorageType m_data {};
	};

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res += rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res -= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res *= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res /= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res += rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res -= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res *= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res /= rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res += rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res -= rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res *= rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res /= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE Scalar dist2(const VecImpl<Scalar, Dims, StorageType> &lhs,
								 const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return (lhs - rhs).mag2();
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE Scalar dist(const VecImpl<Scalar, Dims, StorageType> &lhs,
								const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return (lhs - rhs).mag();
	}

	template<typename Scalar, i64 Dims>
	using Vec = VecImpl<Scalar, Dims, Vc::SimdArray<Scalar, Dims>>;
} // namespace librapid

#ifdef FMT_API
template<typename Scalar, librapid::i64 D, typename StorageType>
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

#endif // LIBRAPID_VECTOR
