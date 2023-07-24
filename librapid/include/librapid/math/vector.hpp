#ifndef LIBRAPID_MATH_VECTOR_OLD_HPP
#define LIBRAPID_MATH_VECTOR_OLD_HPP

// #include "genericVector.hpp"
// #include "simdVector.hpp"
#include "vectorImpl.hpp"

namespace librapid {
//	namespace detail {
//		template<typename Scalar, int64_t Dims>
//		auto vectorTypeIdentifier() {
//			if constexpr (typetraits::TypeInfo<Scalar>::packetWidth > 1) {
//				return SIMDVector<Scalar, Dims> {};
//			} else {
//				return GenericVector<Scalar, Dims> {};
//			}
//		}
//	} // namespace detail
//
//	/// A simplified interface to the GenericVector class, defaulting to Vc SimdArray storage
//	/// \tparam Scalar The scalar type of the vector
//	/// \tparam Dims The dimensionality of the vector
//	template<typename Scalar, int64_t Dims>
//	using Vec = decltype(detail::vectorTypeIdentifier<Scalar, Dims>());
//
//	using Vec2i = Vec<int32_t, 2>;
//	using Vec3i = Vec<int32_t, 3>;
//	using Vec4i = Vec<int32_t, 4>;
//	using Vec2f = Vec<float, 2>;
//	using Vec3f = Vec<float, 3>;
//	using Vec4f = Vec<float, 4>;
//	using Vec2d = Vec<double, 2>;
//	using Vec3d = Vec<double, 3>;
//	using Vec4d = Vec<double, 4>;
//
//	using Vec2 = Vec2d;
//	using Vec3 = Vec3d;
//	using Vec4 = Vec4d;

	using Vec2i = Vector<int32_t, 2>;
	using Vec3i = Vector<int32_t, 3>;
	using Vec4i = Vector<int32_t, 4>;
	using Vec2f = Vector<float, 2>;
	using Vec3f = Vector<float, 3>;
	using Vec4f = Vector<float, 4>;
	using Vec2d = Vector<double, 2>;
	using Vec3d = Vector<double, 3>;
	using Vec4d = Vector<double, 4>;
} // namespace librapid

#endif // LIBRAPID_MATH_VECTOR_OLD_HPP