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
		explicit Vec(X x, YZ... yz) : m_components {(DTYPE)x, (DTYPE)yz...} {
			static_assert(sizeof...(YZ) <= dims, "Parameters cannot exceed vector dimensions");
		}

		template<typename T>
		Vec(const std::initializer_list<T> &vals) {
			LR_ASSERT(vals.size() <= dims, "Parameters cannot exceed vector dimensions");
			i64 ind = 0;
			for (const auto &val : vals) { m_components[ind++] = val; }
		}

		template<typename T, i64 d>
		explicit Vec(const Vec<T, d> &other) {
			i64 i;
			for (i = 0; i < MIN_DIM_CLAMP(dims, d); ++i) { m_components[i] = other[i]; }
		}

		template<typename T>
		Vec(const Vec<T, 3> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec(const Vec<DTYPE, dims> &other) {
			i64 i;
			for (i = 0; i < dims; ++i) { m_components[i] = other[i]; }
		}

		Vec<DTYPE, dims> &operator=(const Vec<DTYPE, dims> &other) {
			if (this == &other) { return *this; }
			for (i64 i = 0; i < dims; ++i) { m_components[i] = other[i]; }
			return *this;
		}

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<tmpDim, T, p> &vec) {
			for (i64 i = 0; i < tmpDim; ++i) { m_components[i] = (i < dims) ? ((T)vec[i]) : (T()); }
		}

		template<typename T = DTYPE, int tmpDim = dims, glm::qualifier p = glm::defaultp>
		operator glm::vec<tmpDim, T, p>() const {
			glm::vec<tmpDim, T, p> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = (i < dims) ? ((T)m_components[i]) : (T()); }
			return res;
		}

#endif // GLM_VERSION

		// Implement indexing (const and non-const)
		// Functions take a single index and return a scalar value

		const DTYPE &operator[](i64 index) const { return m_components[index]; }

		DTYPE &operator[](i64 index) { return m_components[index]; }

		template<typename T, i64 tmpDims>
		bool operator==(const Vec<T, tmpDims> &other) const {
			// For vectors with different dimensions, return true if the excess
			// values are all zero
			for (i64 i = 0; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				if (m_components[i] != other[i]) return false;
			}

			// Quick return to avoid excess checks
			if (dims == tmpDims) return true;

			for (i64 i = MIN_DIM_CLAMP(dims, tmpDims); i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims && m_components[i]) return false;
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
				if (m_components[i] != other[i]) return false;
			}

			// Quick return to avoid excess checks
			if (dims == tmpDims) return true;

			for (i64 i = MIN_DIM_CLAMP(dims, tmpDims); i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims && m_components[i]) return false;
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
			for (i64 i = 0; i < dims; ++i) { res[i] = -m_components[i]; }
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
				res[i] = ((i < dims) ? m_components[i] : 0) + ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) - ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) * ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (i64 i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) / ((i < tmpDims) ? other[i] : 0);
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
			for (i64 i = 0; i < dims; ++i) { res[i] = m_components[i] + other; }
			return res;
		}

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator-(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_components[i] - other; }
			return res;
		}

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator*(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_components[i] * other; }
			return res;
		}

		template<typename T, typename std::enable_if<internal::traits<T>::IsScalar, int>::type = 0>
		Vec<Common<T>, dims> operator/(const T &other) const {
			Vec<Common<T>, dims> res;
			for (i64 i = 0; i < dims; ++i) { res[i] = m_components[i] / other; }
			return res;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator+=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] += (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator-=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] -= (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator*=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] *= (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T, i64 tmpDims>
		Vec<DTYPE, dims> &operator/=(const Vec<T, tmpDims> &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] /= (i < tmpDims) ? (other[i]) : (0); }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator+=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] += other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator-=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] -= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator*=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] *= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator/=(const T &other) {
			for (i64 i = 0; i < dims; ++i) { m_components[i] /= other; }
			return *this;
		}

		//
		// Return the magnitude squared of a vector
		//
		DTYPE mag2() const {
			DTYPE res = 0;
			for (i64 i = 0; i < dims; ++i) { res += m_components[i] * m_components[i]; }
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
				squared += (m_components[i] - other[i]) * (m_components[i] - other[i]);
			}

			// Compute the squares of the values for the remaining values.
			// This just enables calculating the distance between two vectors
			// with different dimensions
			for (; i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims)
					squared += m_components[i] * m_components[i];
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
		Common<T> dot(const Vec<T, dims> &other) const {
			Common<T> res = 0;
			for (i64 i = 0; i < dims; ++i) { res += m_components[i] * other[i]; }
			return res;
		}

		//
		// Compute the vector cross product
		//
		template<typename T>
		Vec<Common<T>, dims> cross(const Vec<T, dims> &other) const {
			static_assert(dims == 2 || dims == 3,
						  "Only 2D and 3D vectors support the cross product");

			Vec<Common<T>, dims> res;

			if constexpr (dims == 2) {
				m_components[2] = 0;
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
				res += ::librapid::str(m_components[i]) + (i == dims - 1 ? ")" : ", ");
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

		DTYPE &x = m_components[0];
		DTYPE &y = m_components[1];
		DTYPE &z = m_components[2];
		DTYPE &w = m_components[3];

	private:
		DTYPE m_components[dims < 4 ? 4 : dims];
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

	template<typename T, i64 dims>
	std::ostream &operator<<(std::ostream &os, const Vec<T, dims> &vec) {
		return os << vec.str();
	}
	*/

	template<typename T, i64 dims>
	class Vec {
	public:
		using StorageType = Vc::SimdArray<T, dims>;

		Vec() = default;

		Vec(const StorageType &arr) : m_components {+arr} {}

		template<typename... Args>
		Vec(Args... args) : m_components {args...} {
			static_assert(sizeof...(Args) <= dims, "Invalid number of arguments");
		}

		Vec(const Vec &other)			 = default;
		Vec(Vec &&other)				 = default;
		Vec &operator=(const Vec &other) = default;
		Vec &operator=(Vec &&other)		 = default;

		LR_NODISCARD("") auto operator[](i64 index) const { return m_components[index]; }

		LR_NODISCARD("") auto &operator[](i64 index) { return m_components[index]; }

		LR_FORCE_INLINE void operator+=(const Vec &other) { m_components += other.m_components; }
		LR_FORCE_INLINE void operator-=(const Vec &other) { m_components -= other.m_components; }
		LR_FORCE_INLINE void operator*=(const Vec &other) { m_components *= other.m_components; }
		LR_FORCE_INLINE void operator/=(const Vec &other) { m_components /= other.m_components; }

		LR_FORCE_INLINE void operator+=(const T &value) { m_components += value; }
		LR_FORCE_INLINE void operator-=(const T &value) { m_components -= value; }
		LR_FORCE_INLINE void operator*=(const T &value) { m_components *= value; }
		LR_FORCE_INLINE void operator/=(const T &value) { m_components /= value; }

		LR_NODISCARD("") LR_INLINE T mag2() const { return (m_components * m_components).sum(); }
		LR_NODISCARD("") LR_INLINE T mag() const { return ::librapid::sqrt(mag2()); }

		LR_NODISCARD("") std::string str() const {
			std::string res = "(";
			for (i64 i = 0; i < dims; ++i) {
				res += std::to_string(m_components[i]);
				if (i != dims - 1) { res += ", "; }
			}
			return res + ")";
		}

	protected:
		StorageType m_components {};
	};

	template<typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator+(const Vec<T, dims> &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(lhs);
		res += rhs;
		return res;
	}

	template<typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator-(const Vec<T, dims> &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(lhs);
		res -= rhs;
		return res;
	}

	template<typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator*(const Vec<T, dims> &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(lhs);
		res *= rhs;
		return res;
	}

	template<typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator/(const Vec<T, dims> &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(lhs);
		res /= rhs;
		return res;
	}

	template<typename T, i64 dims, typename S>
	LR_FORCE_INLINE Vec<T, dims> operator+(const Vec<T, dims> &lhs, const S &rhs) {
		Vec<T, dims> res(lhs);
		res += rhs;
		return res;
	}

	template<typename T, i64 dims, typename S>
	LR_FORCE_INLINE Vec<T, dims> operator-(const Vec<T, dims> &lhs, const S &rhs) {
		Vec<T, dims> res(lhs);
		res -= rhs;
		return res;
	}

	template<typename T, i64 dims, typename S>
	LR_FORCE_INLINE Vec<T, dims> operator*(const Vec<T, dims> &lhs, const S &rhs) {
		Vec<T, dims> res(lhs);
		res *= rhs;
		return res;
	}

	template<typename T, i64 dims, typename S>
	LR_FORCE_INLINE Vec<T, dims> operator/(const Vec<T, dims> &lhs, const S &rhs) {
		Vec<T, dims> res(lhs);
		res /= rhs;
		return res;
	}

	template<typename S, typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator+(const S &lhs, const Vec<T, dims> &rhs) {
		using StorageType = typename Vec<T, dims>::StorageType;
		Vec<T, dims> res(StorageType(static_cast<T>(lhs)));
		res += rhs;
		return res;
	}

	template<typename S, typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator-(const S &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(StorageType(static_cast<T>(lhs)));
		res -= rhs;
		return res;
	}

	template<typename S, typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator*(const S &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(StorageType(static_cast<T>(lhs)));
		res *= rhs;
		return res;
	}

	template<typename S, typename T, i64 dims>
	LR_FORCE_INLINE Vec<T, dims> operator/(const S &lhs, const Vec<T, dims> &rhs) {
		Vec<T, dims> res(StorageType(static_cast<T>(lhs)));
		res /= rhs;
		return res;
	}

} // namespace librapid

#ifdef FMT_API
template<typename T, librapid::i64 D>
struct fmt::formatter<librapid::Vec<T, D>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::Vec<T, D> &arr, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), arr.str());
	}
};
#endif // FMT_API

#endif // LIBRAPID_VECTOR
