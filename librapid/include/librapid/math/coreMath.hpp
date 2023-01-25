#ifndef LIBRAPID_MATH_CORE_MATH_HPP
#define LIBRAPID_MATH_CORE_MATH_HPP

/*
 * This file defines a wide range of core operations on many data types.
 * Many of these functions will end up calling the C++ STL function for
 * primitive types, though for types defined by LibRapid, custom implementations
 * will be required.
 */

namespace librapid {
	namespace detail {
		template<typename First, typename... Types>
		struct ContainsArrayType;
	} // namespace detail

	/// Return the smallest value of a given set of values
	/// \tparam T Data type
	/// \param val Input set
	/// \return Smallest element of the input set
	template<typename T>
	T &&min(T &&val) {
		return std::forward<T>(val);
	}

	/// Return the smallest value of a given set of values
	/// \tparam Types Data types of the input values
	/// \param vals Input values
	/// \return The smallest element of the input values
	template<typename T0, typename T1, typename... Ts>
	auto min(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 < val2) ? min(val1, std::forward<Ts>(vs)...)
							 : min(val2, std::forward<Ts>(vs)...);
	}

	/// Return the largest value of a given set of values
	/// \tparam T Data type
	/// \param val Input set
	/// \return Largest element of the input set
	template<typename T>
	T &&max(T &&val) {
		return std::forward<T>(val);
	}

	/// Return the largest value of a given set of values
	/// \tparam Types Data types of the input values
	/// \param vals Input values
	/// \return The largest element of the input values
	template<typename T0, typename T1, typename... Ts>
	auto max(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 > val2) ? max(val1, std::forward<Ts>(vs)...)
							 : max(val2, std::forward<Ts>(vs)...);
	}

	/// Return the absolute value of a given value
	/// \tparam T Data type
	/// \param val Input value
	/// \return Absolute value of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	constexpr T abs(T val) {
		return std::abs(val);
	}

	/// Map a value from one range to another
	/// \tparam V Data type of the value to map
	/// \tparam B1 Data type of the lower bound of the input range
	/// \tparam E1 Data type of the upper bound of the input range
	/// \tparam B2 Data type of the lower bound of the output range
	/// \tparam E2 Data type of the upper bound of the output range
	/// \param val Value to map
	/// \param start1 Lower bound of the input range
	/// \param stop1 Upper bound of the input range
	/// \param start2 Lower bound of the output range
	/// \param stop2 Upper bound of the output range
	/// \return Mapped value
	template<typename V, typename B1, typename E1, typename B2, typename E2>
	LIBRAPID_INLINE auto map(const V &val, const B1 &start1, const E1 &stop1, const B2 &start2,
							 const E2 &stop2) {
		if constexpr (detail::ContainsArrayType<V, B1, E1, B2, E2>::val) {
			return start2 + (val - start1) * (stop2 - start2) / (stop1 - start1);
		} else {
			using T = decltype((val - start1) * (stop2 - start2) / (stop1 - start1) + start2);
			return static_cast<T>(start2) + (static_cast<T>(stop2) - static_cast<T>(start2)) *
											  ((static_cast<T>(val) - static_cast<T>(start1)) /
											   (static_cast<T>(stop1) - static_cast<T>(start1)));
		}
	}

	template<typename T1, typename T2>
	LIBRAPID_INLINE auto mod(const T1 &val, const T2 &mod) {
		if constexpr (std::is_floating_point_v<T1> || std::is_floating_point_v<T2>) {
			return std::fmod(val, mod);
		} else {
			return val % mod;
		}
	}

	/// Return the floor of a given value
	/// \tparam T Data type
	/// \param val Input value
	/// \return Floor of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr T floor(T val) {
		return std::floor(val);
	}

	/// Return the ceiling of a given value
	/// \tparam T Data type
	/// \param val Input value
	/// \return Ceiling of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr T ceil(T val) {
		return std::ceil(val);
	}

	/// Return the square root of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the square root.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Square root of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto sqrt(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::sqrt(static_cast<double>(val));
		} else {
			return std::sqrt(val);
		}
	}

	/// Return the hypotenuse of a right triangle given the lengths of the two legs. Note that,
	/// for integer values, this function will cast the input values to a floating point type
	/// before calculating the hypotenuse.
	/// \tparam T Data type
	/// \param leg1 Length of the first leg
	/// \param leg2 Length of the second leg
	/// \return Hypotenuse of the right triangle
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto hypot(T leg1, T leg2) {
		if constexpr (std::is_integral_v<T>) {
			return std::hypot(static_cast<double>(leg1), static_cast<double>(leg2));
		} else {
			return std::hypot(leg1, leg2);
		}
	}

	/// Return the cube root of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the cube root.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Cube root of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto cbrt(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::cbrt(static_cast<double>(val));
		} else {
			return std::cbrt(val);
		}
	}

	/// Return the first number raised to the power of the second number. The return value will be
	/// promoted to the larger of the two input types.
	/// \tparam T0 Data type of the first input value
	/// \tparam T1 Data type of the second input value
	/// \param val1 First input value
	/// \param val2 Second input value
	/// \return First input value raised to the power of the second input value
	template<
	  typename T0, typename T1,
	  typename std::enable_if_t<std::is_fundamental_v<T0> && std::is_fundamental_v<T1>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto pow(T0 val1, T1 val2) {
		if constexpr (std::is_integral_v<T0> && std::is_integral_v<T1>) {
			return std::pow(static_cast<double>(val1), static_cast<double>(val2));
		} else if constexpr (std::is_integral_v<T0>) {
			return std::pow(static_cast<double>(val1), val2);
		} else if constexpr (std::is_integral_v<T1>) {
			return std::pow(val1, static_cast<double>(val2));
		} else {
			return std::pow(val1, val2);
		}
	}

	/// Return the exponential of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the exponential.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Exponential of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto exp(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::exp(static_cast<double>(val));
		} else {
			return std::exp(val);
		}
	}

	/// Return 2 raised to a given power. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the exponential.
	/// \tparam T Data type
	/// \param val Input value
	/// \return 2 raised to the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto exp2(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::exp2(static_cast<double>(val));
		} else {
			return std::exp2(val);
		}
	}

	/// Return the natural logarithm of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the logarithm.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Natural logarithm of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto log(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::log(static_cast<double>(val));
		} else {
			return std::log(val);
		}
	}

	/// Return the logarithm base-10 of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the logarithm.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Logarithm base-10 of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto log10(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::log10(static_cast<double>(val));
		} else {
			return std::log10(val);
		}
	}

	/// Return the logarithm base-2 of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the logarithm.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Logarithm base-2 of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto log2(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::log2(static_cast<double>(val));
		} else {
			return std::log2(val);
		}
	}

	/// Return the sine of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the sine.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Sine of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto sin(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::sin(static_cast<double>(val));
		} else {
			return std::sin(val);
		}
	}

	/// Return the cosine of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the cosine.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Cosine of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto cos(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::cos(static_cast<double>(val));
		} else {
			return std::cos(val);
		}
	}

	/// Return the tangent of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the tangent.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Tangent of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto tan(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::tan(static_cast<double>(val));
		} else {
			return std::tan(val);
		}
	}

	/// Return the arcsine of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the arcsine.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Arcsine of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto asin(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::asin(static_cast<double>(val));
		} else {
			return std::asin(val);
		}
	}

	/// Return the arccosine of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the arccosine.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Arccosine of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto acos(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::acos(static_cast<double>(val));
		} else {
			return std::acos(val);
		}
	}

	/// Return the arctangent of a given value. Note that, for integer values, this function
	/// will cast the input value to a floating point type before calculating the arctangent.
	/// \tparam T Data type
	/// \param val Input value
	/// \return Arctangent of the input value
	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr auto atan(T val) {
		if constexpr (std::is_integral_v<T>) {
			return std::atan(static_cast<double>(val));
		} else {
			return std::atan(val);
		}
	}
} // namespace librapid

#endif // LIBRAPID_MATH_CORE_MATH_HPP
