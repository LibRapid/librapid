#ifndef LIBRAPID_MATH_CORE_MATH_HPP
#define LIBRAPID_MATH_CORE_MATH_HPP

/*
 * This file defines a wide range of core operations on many data types.
 * Many of these functions will end up calling the C++ STL function for
 * primitive types, though for types defined by LibRapid, custom implementations
 * will be required.
 */

namespace librapid {
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
} // namespace librapid

#endif // LIBRAPID_MATH_CORE_MATH_HPP
