#ifndef LIBRAPID_VECTOR
#define LIBRAPID_VECTOR

#include "../internal/config.hpp"

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

	template<typename DTYPE, int64_t dims>
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
			int64_t ind = 0;
			for (const auto &val : vals) { m_components[ind++] = val; }
		}

		template<typename T, int64_t d>
		explicit Vec(const Vec<T, d> &other) {
			int64_t i;
			for (i = 0; i < MIN_DIM_CLAMP(dims, d); ++i) { m_components[i] = other[i]; }
		}

		template<typename T>
		Vec(const Vec<T, 3> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec(const Vec<DTYPE, dims> &other) {
			int64_t i;
			for (i = 0; i < dims; ++i) { m_components[i] = other[i]; }
		}

		Vec<DTYPE, dims> &operator=(const Vec<DTYPE, dims> &other) {
			if (this == &other) { return *this; }
			for (int64_t i = 0; i < dims; ++i) { m_components[i] = other[i]; }
			return *this;
		}

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<tmpDim, T, p> &vec) {
			for (int64_t i = 0; i < tmpDim; ++i) {
				m_components[i] = (i < dims) ? ((T)vec[i]) : (T());
			}
		}

		template<typename T = DTYPE, int tmpDim = dims, glm::qualifier p = glm::defaultp>
		operator glm::vec<tmpDim, T, p>() const {
			glm::vec<tmpDim, T, p> res;
			for (int64_t i = 0; i < dims; ++i) {
				res[i] = (i < dims) ? ((T)m_components[i]) : (T());
			}
			return res;
		}

#endif // GLM_VERSION

		/**
		 * Implement indexing (const and non-const)
		 * Functions take a single index and return a scalar value
		 */

		const DTYPE &operator[](int64_t index) const { return m_components[index]; }

		DTYPE &operator[](int64_t index) { return m_components[index]; }

		template<typename T, int64_t tmpDims>
		bool operator==(const Vec<T, tmpDims> &other) const {
			// For vectors with different dimensions, return true if the excess
			// values are all zero
			for (int64_t i = 0; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				if (m_components[i] != other[i]) return false;
			}

			// Quick return to avoid excess checks
			if (dims == tmpDims) return true;

			for (int64_t i = MIN_DIM_CLAMP(dims, tmpDims); i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims && m_components[i]) return false;
				if (i < tmpDims && other[i]) return false;
			}

			return true;
		}

		template<typename T, int64_t tmpDims>
		bool operator!=(const Vec<T, tmpDims> &other) const {
			return !(*this == other);
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take two Vec objects and return a new vector (with common
		 * type) containing the result of the element-wise operation.
		 *
		 * Vectors must have same dimensions. To cast, use Vec.as<TYPE, DIMS>()
		 */
		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) + ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) - ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) * ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) / ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take a vector and a scalar, and return a new vector (with
		 * common type) containing the result of the element-wise operation.
		 */

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator+(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] + other; }
			return res;
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator-(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] - other; }
			return res;
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator*(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] * other; }
			return res;
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator/(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] / other; }
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator+=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] += (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator-=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] -= (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator*=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] *= (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator/=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] /= (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator+=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] += other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator-=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] -= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator*=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] *= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator/=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] /= other; }
			return *this;
		}

		/**
		 * Return the magnitude squared of a vector
		 */
		DTYPE mag2() const {
			DTYPE res = 0;
			for (int64_t i = 0; i < dims; ++i) { res += m_components[i] * m_components[i]; }
			return res;
		}

		/**
		 * Return the magnitude of a vector
		 */
		DTYPE mag() const { return sqrt(mag2()); }

		DTYPE invMag() const { return DTYPE(1) / sqrt(mag2()); }

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist2(const Vec<T, tmpDims> &other) const {
			using RET	= typename std::common_type<DTYPE, T>::type;
			RET squared = 0;
			int64_t i	= 0;

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

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist(const Vec<T, tmpDims> &other) const {
			return sqrt(dist2(other));
		}

		/**
		 * Compute the vector dot product
		 * AxBx + AyBy + AzCz + ...
		 */
		template<typename T>
		Common<T> dot(const Vec<T, dims> &other) const {
			Common<T> res = 0;
			for (int64_t i = 0; i < dims; ++i) { res += m_components[i] * other[i]; }
			return res;
		}

		/**
		 * Compute the vector cross product
		 */
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
			for (int64_t i = 0; i < dims; ++i) {
				res += std::to_string(m_components[i]) + (i == dims - 1 ? ")" : ", ");
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

	/**
	 * Implement simple arithmetic operators + - * /
	 *
	 * Operations take a scalar and a vector and return a new vector (with
	 * common type) containing the result of the element-wise operation.
	 */

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator+(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value + vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator-(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value - vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator*(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value * vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator/(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value / vec[i]; }
		return res;
	}

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	template<typename DTYPE>
	class Vec<DTYPE, 3> {
		template<typename T>
		using Common = typename std::common_type<DTYPE, T>::type;

	public:
		Vec() = default;

		template<typename X = DTYPE, typename Y = DTYPE, typename Z = DTYPE>
		explicit Vec(X x, Y y = 0, Z z = 0) : x(x), y(y), z(z) {}

		template<typename T>
		Vec(const std::initializer_list<T> &vals) {
			LR_ASSERT(vals.size() <= 3, "Parameters cannot exceed vector dimensions");
			int64_t ind = 0;
			for (const auto &val : vals) { operator[](ind++) = val; }
		}

		template<typename T, int64_t d>
		explicit Vec(const Vec<T, d> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec(const Vec<DTYPE, 3> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec<DTYPE, 3> &operator=(const Vec<DTYPE, 3> &other) {
			if (this == &other) { return *this; }
			x = other.x;
			y = other.y;
			z = other.z;
			return *this;
		}

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<typename T, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<2, T, p> &vec) {
			x = vec.x;
			y = vec.y;
		}

		template<typename T, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<3, T, p> &vec) {
			x = vec.x;
			y = vec.y;
			z = vec.z;
		}

		template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
		operator glm::vec<tmpDim, T, p>() const {
			glm::vec<tmpDim, T, p> res;
			for (int64_t i = 0; i < tmpDim; ++i) { res[i] = (i < 3) ? ((&x)[i]) : (T(0)); }
			return res;
		}

#endif // GLM_VERSION

		/**
		 * Implement indexing (const and non-const)
		 * Functions take a single index and return a scalar value
		 */

		const DTYPE &operator[](int64_t index) const { return (&x)[index]; }

		DTYPE &operator[](int64_t index) { return (&x)[index]; }

		template<typename T, int64_t tmpDims>
		bool operator==(const Vec<T, tmpDims> &other) const {
			if (tmpDims <= 3) {
				return x == other.x && y == other.y && z == other.z && w == other.w;
			}

			for (int64_t i = 3; i < tmpDims; ++i) {
				if (other[i]) return false;
			}

			return true;
		}

		template<typename T, int64_t tmpDims>
		bool operator!=(const Vec<T, tmpDims> &other) const {
			return !(*this == other);
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take two Vec objects and return a new vector (with
		 * common type) containing the result of the element-wise operation.
		 *
		 * Vectors must have same dimensions. To cast, use Vec.as<TYPE,
		 * DIMS>()
		 */
		template<typename T>
		Vec<Common<T>, 3> operator+(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x + other.x, y + other.y, z + other.z);
		}

		template<typename T>
		Vec<Common<T>, 3> operator-(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x - other.x, y - other.y, z - other.z);
		}

		template<typename T>
		Vec<Common<T>, 3> operator*(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x * other.x, y * other.y, z * other.z);
		}

		template<typename T>
		Vec<Common<T>, 3> operator/(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x / other.x, y / other.y, z / other.z);
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) + ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) - ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) * ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) / ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take a vector and a scalar, and return a new vector
		 * (with common type) containing the result of the element-wise
		 * operation.
		 */

		template<typename T>
		Vec<Common<T>, 3> operator+(const T &other) const {
			return Vec<Common<T>, 3>(x + other, y + other, z + other);
		}

		template<typename T>
		Vec<Common<T>, 3> operator-(const T &other) const {
			return Vec<Common<T>, 3>(x - other, y - other, z - other);
		}

		template<typename T>
		Vec<Common<T>, 3> operator*(const T &other) const {
			return Vec<Common<T>, 3>(x * other, y * other, z * other);
		}

		template<typename T>
		Vec<Common<T>, 3> operator/(const T &other) const {
			return Vec<Common<T>, 3>(x / other, y / other, z / other);
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator+=(const Vec<T, tmpDims> &other) {
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator-=(const Vec<T, tmpDims> &other) {
			x -= other.x;
			y -= other.y;
			z -= other.z;
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator*=(const Vec<T, tmpDims> &other) {
			x *= other.x;
			y *= other.y;
			z *= other.z;
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator/=(const Vec<T, tmpDims> &other) {
			x /= other.x;
			y /= other.y;
			z /= other.z;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator+=(const T &other) {
			x += other;
			y += other;
			z += other;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator-=(const T &other) {
			x -= other;
			y -= other;
			z -= other;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator*=(const T &other) {
			x *= other;
			y *= other;
			z *= other;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator/=(const T &other) {
			x /= other;
			y /= other;
			z /= other;
			return *this;
		}

		/**
		 * Return the magnitude squared of a vector
		 */
		DTYPE mag2() const { return x * x + y * y + z * z; }

		/**
		 * Return the magnitude of a vector
		 */
		DTYPE mag() const { return sqrt(x * x + y * y + z * z); }

		DTYPE invMag() const {
			DTYPE mag = x * x + y * y + z * z;
			return DTYPE(1) / mag;
		}

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist2(const Vec<T, tmpDims> &other) const {
			// Specific case for a 2D vector
			if constexpr (tmpDims == 2) {
				return ((x - other.x) * (x - other.x)) + ((y - other.y) * (y - other.y)) + (z * z);
			}

			// Specific case for a 3D vector
			if constexpr (tmpDims == 3) {
				return ((x - other.x) * (x - other.x)) + ((y - other.y) * (y - other.y)) +
					   ((z - other.z) * (z - other.z));
			}

			// Specific case for a 4D vector
			if constexpr (tmpDims == 4) {
				return ((x - other.x) * (x - other.x)) + ((y - other.y) * (y - other.y)) +
					   ((z - other.z) * (z - other.z)) + (other.w * other.w);
			}

			// General case for 1, 5, 6, 7, ... dimensional vectors
			using RET	= typename std::common_type<DTYPE, T>::type;
			RET squared = 0;
			int64_t i	= 0;

			// Compute the squares of the differences for the matching
			// components
			for (; i < MIN_DIM_CLAMP(3, tmpDims); ++i) {
				squared += ((&x)[i] - other[i]) * ((&x)[i] - other[i]);
			}

			// Compute the squares of the values for the remaining values.
			// This just enables calculating the distance between two vectors
			// with different dimensions
			for (; i < MAX_DIM_CLAMP(3, tmpDims); ++i) {
				if (i < 3)
					squared += (&x)[i] * (&x)[i];
				else
					squared += other[i] * other[i];
			}
		}

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist(const Vec<T, tmpDims> &other) const {
			return sqrt(dist2(other));
		}

		/**
		 * Compute the vector dot product
		 * AxBx + AyBy + AzCz + ...
		 */
		template<typename T>
		Common<T> dot(const Vec<T, 3> &other) const {
			return x * other.x + y * other.y + z * other.z;
		}

		/**
		 * Compute the vector cross product
		 */
		template<typename T>
		Vec<Common<T>, 3> cross(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(
			  y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
		}

		// Swizzle Operations

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
			return std::string("(") + std::to_string(x) + ", " + std::to_string(y) + ", " +
				   std::to_string(z) + ")";
		}

		void setX(DTYPE val) { x = val; }

		void setY(DTYPE val) { y = val; }

		void setZ(DTYPE val) { z = val; }

		void setW(DTYPE val) { w = val; }

		DTYPE getX() { return x; }

		DTYPE getY() { return y; }

		DTYPE getZ() { return z; }

		DTYPE getW() { return w; }

		DTYPE x = 0;
		DTYPE y = 0;
		DTYPE z = 0;
		DTYPE w = 0;
	};

	/**
	 * Implement simple arithmetic operators + - * /
	 *
	 * Operations take a scalar and a vector and return a new vector (with
	 * common type) containing the result of the element-wise operation.
	 */

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator+(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value + vec.x, value + vec.y, value + vec.z);
	}

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator-(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value - vec.x, value - vec.y, value - vec.z);
	}

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator*(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value * vec.x, value * vec.y, value * vec.z);
	}

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator/(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value / vec.x, value / vec.y, value / vec.z);
	}

	using Vec2i = Vec<int64_t, 2>;
	using Vec2f = Vec<float, 2>;
	using Vec2d = Vec<double, 2>;

	using Vec3i = Vec<int64_t, 3>;
	using Vec3f = Vec<float, 3>;
	using Vec3d = Vec<double, 3>;

	using Vec4i = Vec<int64_t, 4>;
	using Vec4f = Vec<float, 4>;
	using Vec4d = Vec<double, 4>;

	template<typename T, int64_t dims>
	std::ostream &operator<<(std::ostream &os, const Vec<T, dims> &vec) {
		return os << vec.str();
	}
} // namespace librapid

#ifdef FMT_API
template<typename T, int64_t D>
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
