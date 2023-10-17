#ifndef LIBRAPID_MATH_VECTOR_OLD_HPP
#define LIBRAPID_MATH_VECTOR_OLD_HPP

#include "vectorForward.hpp"
#include "vectorImpl.hpp"

namespace librapid {
    using Vec2i = Vector<int32_t, 2>;
    using Vec3i = Vector<int32_t, 3>;
    using Vec4i = Vector<int32_t, 4>;
    using Vec2f = Vector<float, 2>;
    using Vec3f = Vector<float, 3>;
    using Vec4f = Vector<float, 4>;
    using Vec2d = Vector<double, 2>;
    using Vec3d = Vector<double, 3>;
    using Vec4d = Vector<double, 4>;

    using Vec2 = Vector<double, 2>;
    using Vec3 = Vector<double, 3>;
    using Vec4 = Vector<double, 4>;

    template<typename T = float, uint64_t numDims = 3>
    Vector<T, numDims> pointOnUnitSphere() {
        // Given X = [x_1, x_2, ..., x_n], where x_n ~ N(1, 0),
        // X / |X| will be uniformly distributed on the unit sphere

        Vector<T, numDims> result;
        for (uint64_t i = 0; i < numDims; ++i) {
            result[i] = ::librapid::randomGaussian();
        }

        return result.norm();
    }

    /// \brief Generate a random point within the unit sphere
    /// \tparam T Scalar type of the vector, defaulting to ``float``
    /// \return 3D vector, \f$\vec{v}\f$, with \f$0 \leq |\vec{v}| \leq 1\f$
    template<typename T = float>
    Vector<T, 3> randomPointInSphere() {
        // Adapted from https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/

        auto u = lrc::random<T>(); // [0, 1)
        auto v = lrc::random<T>(); // [0, 1)
        auto theta = u * 2 * PI;
        auto phi = ::librapid::acos(2 * v - 1);
        auto r = ::librapid::cbrt(random<T>());
        auto sinTheta = ::librapid::sin(theta);
        auto cosTheta = ::librapid::cos(theta);
        auto sinPhi = ::librapid::sin(phi);
        auto cosPhi = ::librapid::cos(phi);
        auto x = r * sinPhi * cosTheta;
        auto y = r * sinPhi * sintheta;
        auto z = r * cosPhi;
        return {x, y, z};
    }
} // namespace librapid

#endif // LIBRAPID_MATH_VECTOR_OLD_HPP