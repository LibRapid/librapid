#ifndef LIBRAPID_VECTOR
#define LIBRAPID_VECTOR

#include <librapid/config.hpp>

#include <cstdint>
#include <cstring>
#include <ostream>
#include <type_traits>

namespace librapid {
    template<typename DTYPE, int64_t dims>
    class Vec {
        template<typename T>
        using Common = typename std::common_type<DTYPE, T>::type;

    public:
        Vec() = default;;

        template<typename X, typename ...YZ>
        Vec(X x, YZ ... yz) : m_components{(DTYPE) x, (DTYPE) yz...} {
            static_assert(1 + sizeof...(YZ) <= dims, "Parameters cannot exceed vector dimensions");
        }

        Vec(const Vec<DTYPE, dims> &other) {
            int64_t i;
            for (i = 0; i < dims; ++i) m_components[i] = other.m_components[i];
        }

        Vec<DTYPE, dims> &operator=(const Vec<DTYPE, dims> &other) {
            if (this == &other) return *this;
            for (int64_t i = 0; i < dims; ++i) m_components[i] = other.m_components[i];
            return *this;
        }

        // Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

        template<glm::qualifier p>
        Vec(const glm::vec<dims, DTYPE, p> &vec) {
            for (int64_t i = 0; i < dims; ++i)
                m_components[i] = vec[i];
        }

        template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
        operator glm::vec<tmpDim, T, p>() const {
            glm::vec<tmpDim, T, p> res;
            for (int64_t i = 0; i < dims; ++i)
                res[i] = (i < dims) ? ((T) m_components[i]) : (T());
            return res;
        }

#endif // GLM_VERSION

        /**
         * Implement indexing (const and non-const)
         * Functions take a single index and return a scalar value
         */

        const DTYPE &operator[](int64_t index) const { return m_components[index]; }

        DTYPE &operator[](int64_t index) { return m_components[index]; }

        /**
         * Implement simple arithmetic operators + - * /
         *
         * Operations take two Vec objects and return a new vector (with common type)
         * containing the result of the element-wise operation.
         *
         * Vectors must have same dimensions. To cast, use Vec.as<TYPE, DIMS>()
         */
        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> res;
            for (int64_t i = 0; i < ((dims > tmpDims) ? (dims) : (tmpDims)); ++i)
                res[i] = ((i < dims) ? m_components[i] : 0) + ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> res;
            for (int64_t i = 0; i < ((dims > tmpDims) ? (dims) : (tmpDims)); ++i)
                res[i] = ((i < dims) ? m_components[i] : 0) - ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> res;
            for (int64_t i = 0; i < ((dims > tmpDims) ? (dims) : (tmpDims)); ++i)
                res[i] = ((i < dims) ? m_components[i] : 0) * ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (dims > tmpDims) ? (dims) : (tmpDims)> res;
            for (int64_t i = 0; i < ((dims > tmpDims) ? (dims) : (tmpDims)); ++i)
                res[i] = ((i < dims) ? m_components[i] : 0) / ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        /**
         * Implement simple arithmetic operators + - * /
         *
         * Operations take a vector and a scalar, and return a new vector (with common type)
         * containing the result of the element-wise operation.
         */

        template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
        Vec<Common<T>, dims> operator+(const T &other) const {
            Vec<Common<T>, dims> res;
            for (int64_t i = 0; i < dims; ++i) res[i] = m_components[i] + other;
            return res;
        }

        template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
        Vec<Common<T>, dims> operator-(const T &other) const {
            Vec<Common<T>, dims> res;
            for (int64_t i = 0; i < dims; ++i) res[i] = m_components[i] - other;
            return res;
        }

        template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
        Vec<Common<T>, dims> operator*(const T &other) const {
            Vec<Common<T>, dims> res;
            for (int64_t i = 0; i < dims; ++i) res[i] = m_components[i] * other;
            return res;
        }

        template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
        Vec<Common<T>, dims> operator/(const T &other) const {
            Vec<Common<T>, dims> res;
            for (int64_t i = 0; i < dims; ++i) res[i] = m_components[i] / other;
            return res;
        }


        template<typename T, int64_t tmpDims>
        Vec<DTYPE, dims> &operator+=(const Vec<T, tmpDims> &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] += (i < tmpDims) ? (other[i]) : (0);
            return *this;
        }

        template<typename T, int64_t tmpDims>
        Vec<DTYPE, dims> &operator-=(const Vec<T, tmpDims> &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] -= (i < tmpDims) ? (other[i]) : (0);
            return *this;
        }

        template<typename T, int64_t tmpDims>
        Vec<DTYPE, dims> &operator*=(const Vec<T, tmpDims> &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] *= (i < tmpDims) ? (other[i]) : (0);
            return *this;
        }

        template<typename T, int64_t tmpDims>
        Vec<DTYPE, dims> &operator/=(const Vec<T, tmpDims> &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] /= (i < tmpDims) ? (other[i]) : (0);
            return *this;
        }


        template<typename T>
        Vec<DTYPE, dims> &operator+=(const T &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] += other;
            return *this;
        }

        template<typename T>
        Vec<DTYPE, dims> &operator-=(const T &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] -= other;
            return *this;
        }

        template<typename T>
        Vec<DTYPE, dims> &operator*=(const T &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] *= other;
            return *this;
        }

        template<typename T>
        Vec<DTYPE, dims> &operator/=(const T &other) {
            for (int64_t i = 0; i < dims; ++i) m_components[i] /= other;
            return *this;
        }

        /**
         * Return the magnitude squared of a vector
         */
        DTYPE mag2() const {
            DTYPE res = 0;
            for (const auto &val: m_components) res += val * val;
            return res;
        }

        /**
         * Return the magnitude of a vector
         */
        DTYPE mag() const {
            return sqrt(mag2());
        }

        DTYPE invMag() const {
            return DTYPE(1) / sqrt(mag2());
        }

        /**
         * Compute the vector dot product
         * AxBx + AyBy + AzCz + ...
         */
        template<typename T>
        Common<T> dot(const Vec<T, dims> &other) const {
            Common<T> res = 0;
            for (int64_t i = 0; i < dims; ++i) res += m_components[i] * other[i];
            return res;
        }

        /**
         * Compute the vector cross product
         */
        template<typename T>
        Vec<Common<T>, dims> cross(const Vec<T, dims> &other) const {
            static_assert(dims == 2 || dims == 3, "Only 2D and 3D vectors support the cross product");

            Vec<Common<T>, dims> res;

            if constexpr (dims == 2) {
                m_components[2] = 0;
                other[2] = 0;
            }

            res.x = y * other.z - z * other.y;
            res.y = z * other.x - x * other.z;
            res.z = x * other.y - y * other.x;

            return res;
        }

        [[nodiscard]] std::string str() const {
            std::string res = "(";
            for (int64_t i = 0; i < dims; ++i) res += std::to_string(m_components[i]) + (i == dims - 1 ? ")" : ", ");
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
     * Operations take a scalar and a vector and return a new vector (with common type)
     * containing the result of the element-wise operation.
     */

    template<typename T, typename DTYPE, int64_t dims, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, dims> operator+(const T &value, const Vec<DTYPE, dims> &vec) {
        Vec<typename std::common_type<T, DTYPE>::type, dims> res;
        for (int64_t i = 0; i < dims; ++i) res[i] = value + vec[i];
        return res;
    }

    template<typename T, typename DTYPE, int64_t dims, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, dims> operator-(const T &value, const Vec<DTYPE, dims> &vec) {
        Vec<typename std::common_type<T, DTYPE>::type, dims> res;
        for (int64_t i = 0; i < dims; ++i) res[i] = value - vec[i];
        return res;
    }

    template<typename T, typename DTYPE, int64_t dims, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, dims> operator*(const T &value, const Vec<DTYPE, dims> &vec) {
        Vec<typename std::common_type<T, DTYPE>::type, dims> res;
        for (int64_t i = 0; i < dims; ++i) res[i] = value * vec[i];
        return res;
    }

    template<typename T, typename DTYPE, int64_t dims, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, dims> operator/(const T &value, const Vec<DTYPE, dims> &vec) {
        Vec<typename std::common_type<T, DTYPE>::type, dims> res;
        for (int64_t i = 0; i < dims; ++i) res[i] = value / vec[i];
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
        Vec() = default;;

        template<typename X = DTYPE, typename Y = DTYPE, typename Z = DTYPE>
        Vec(X x, Y y = 0, Z z = 0) : x(x), y(y), z(z) {}

        Vec(const Vec<DTYPE, 3> &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        Vec<DTYPE, 3> &operator=(const Vec<DTYPE, 3> &other) {
            if (this == &other) return *this;
            x = other.x;
            y = other.y;
            z = other.z;
            return *this;
        }

        // Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

        template<glm::qualifier p>
        Vec(const glm::vec<3, DTYPE, p> &vec) {
            x = vec.x;
            y = vec.y;
            z = vec.z;
        }

        template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
        operator glm::vec<tmpDim, T, p>() const {
            glm::vec<tmpDim, T, p> res;
            for (int64_t i = 0; i < tmpDim; ++i)
                res[i] = (i < 3) ? ((&x)[i]) : (T(0));
            return res;
        }

#endif // GLM_VERSION

        /**
         * Implement indexing (const and non-const)
         * Functions take a single index and return a scalar value
         */

        const DTYPE &operator[](int64_t index) const { return (&x)[index]; }

        DTYPE &operator[](int64_t index) { return (&x)[index]; }

        /**
         * Implement simple arithmetic operators + - * /
         *
         * Operations take two Vec objects and return a new vector (with common type)
         * containing the result of the element-wise operation.
         *
         * Vectors must have same dimensions. To cast, use Vec.as<TYPE, DIMS>()
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
        Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> res;
            for (int64_t i = 0; i < ((3 > tmpDims) ? (3) : (tmpDims)); ++i)
                res[i] = ((i < 3) ? (&x)[i] : 0) + ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> res;
            for (int64_t i = 0; i < ((3 > tmpDims) ? (3) : (tmpDims)); ++i)
                res[i] = ((i < 3) ? (&x)[i] : 0) - ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> res;
            for (int64_t i = 0; i < ((3 > tmpDims) ? (3) : (tmpDims)); ++i)
                res[i] = ((i < 3) ? (&x)[i] : 0) * ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        template<typename T, int64_t tmpDims>
        Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
            Vec<Common<T>, (3 > tmpDims) ? (3) : (tmpDims)> res;
            for (int64_t i = 0; i < ((3 > tmpDims) ? (3) : (tmpDims)); ++i)
                res[i] = ((i < 3) ? (&x)[i] : 0) / ((i < tmpDims) ? other[i] : 0);
            return res;
        }

        /**
         * Implement simple arithmetic operators + - * /
         *
         * Operations take a vector and a scalar, and return a new vector (with common type)
         * containing the result of the element-wise operation.
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
        DTYPE mag2() const {
            return x * x + y * y + z * z;
        }

        /**
         * Return the magnitude of a vector
         */
        DTYPE mag() const {
            return sqrt(x * x + y * y + z * z);
        }

        DTYPE invMag() const {
            DTYPE mag = x * x + y * y + z * z;
            return DTYPE(1) / mag;
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
                    y * other.z - z * other.y,
                    z * other.x - x * other.z,
                    x * other.y - y * other.x
            );
        }

        [[nodiscard]] std::string str() const {
            return std::string("(") + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
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
 * Operations take a scalar and a vector and return a new vector (with common type)
 * containing the result of the element-wise operation.
 */

    template<typename T, typename DTYPE, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, 3> operator+(const T &value, const Vec<DTYPE, 3> &vec) {
        return Vec<typename std::common_type<T, DTYPE>::type, 3>(value + vec.x, value + vec.y, value + vec.z);
    }

    template<typename T, typename DTYPE, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, 3> operator-(const T &value, const Vec<DTYPE, 3> &vec) {
        return Vec<typename std::common_type<T, DTYPE>::type, 3>(value - vec.x, value - vec.y, value - vec.z);
    }

    template<typename T, typename DTYPE, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, 3> operator*(const T &value, const Vec<DTYPE, 3> &vec) {
        return Vec<typename std::common_type<T, DTYPE>::type, 3>(value * vec.x, value * vec.y, value * vec.z);
    }

    template<typename T, typename DTYPE, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
    Vec<typename std::common_type<T, DTYPE>::type, 3> operator/(const T &value, const Vec<DTYPE, 3> &vec) {
        return Vec<typename std::common_type<T, DTYPE>::type, 3>(value / vec.x, value / vec.y, value / vec.z);
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
}

#endif // LIBRAPID_VECTOR
