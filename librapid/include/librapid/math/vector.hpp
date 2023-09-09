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
} // namespace librapid

#endif // LIBRAPID_MATH_VECTOR_OLD_HPP