#ifndef LIBRAPID_SIMD_TRIGONOMETRY
#define LIBRAPID_SIMD_TRIGONOMETRY

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct IsSIMD : std::false_type {};

		template<typename T, typename U>
		struct IsSIMD<xsimd::batch<T, U>> : std::true_type {};

		template<typename T>
		struct IsSIMD<xsimd::batch_element_reference<T>> : std::true_type {};
	} // namespace typetraits

#define REQUIRE_SIMD(TYPE) typename std::enable_if_t<typetraits::IsSIMD<TYPE>::value, int> = 0
#define IF_FLOATING(TYPE)  if constexpr (std::is_floating_point_v<typename TYPE::value_type>)

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto sin(const T &x) {
		return xsimd::sin(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cos(const T &x) {
		return xsimd::cos(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto tan(const T &x) {
		return xsimd::tan(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto asin(const T &x) {
		return xsimd::asin(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto acos(const T &x) {
		return xsimd::acos(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto atan(const T &x) {
		return xsimd::atan(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto sinh(const T &x) {
		return xsimd::sinh(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cosh(const T &x) {
		return xsimd::cosh(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto tanh(const T &x) {
		return xsimd::tanh(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto exp(const T &x) {
		return xsimd::exp(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto log(const T &x) {
		return xsimd::log(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto log2(const T &x) {
		return xsimd::log2(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto log10(const T &x) {
		return xsimd::log10(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto sqrt(const T &x) {
		return xsimd::sqrt(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto cbrt(const T &x) {
		return xsimd::cbrt(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto abs(const T &x) {
		return xsimd::abs(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto floor(const T &x) {
		return xsimd::floor(x);
	}

	template<typename T, REQUIRE_SIMD(T)>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto ceil(const T &x) {
		return xsimd::ceil(x);
	}
} // namespace librapid

#endif // LIBRAPID_SIMD_TRIGONOMETRY