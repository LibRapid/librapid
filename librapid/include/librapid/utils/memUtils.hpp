#ifndef LIBRAPID_UTILS_MEMUTILS_HPP
#define LIBRAPID_UTILS_MEMUTILS_HPP

namespace librapid {
	/// Cast the bits of one value directly into another type -- no conversion is performed
	/// \tparam To The type to cast to
	/// \tparam From The type to cast from
	/// \param val The value to cast
	/// \return The value bitwise mapped to the new type
	template<typename To, typename From>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE constexpr To bitCast(const From &val) noexcept {
		static_assert(
		  sizeof(To) == sizeof(From),
		  "Types have different sizes, and cannot be cast bit-for-bit between each other");

#if defined(__CUDACC__)
		To toOjb; // assumes default-init
		::std::memcpy(::std::memcpy::addressof(toOjb), ::std::memcpy::addressof(val), sizeof(To));
		return _To_obj;
#elif defined(LIBRAPID_MSVC)
		// MSVC doesn't support std::bit_cast until C++20
		return *(To *)(&val);
#else
		return __builtin_bit_cast(To, val);
#endif
	}

	/// Returns true if the input value is NaN
	/// \tparam T The type of the value
	/// \param val The value to check
	/// \return True if the value is NaN, false otherwise
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isNaN(const T &val) noexcept {
		return std::isnan(val);
	}

	/// Returns true if the input value is finite
	/// \tparam T The type of the value
	/// \param val The value to check
	/// \return True if the value is finite,
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isFinite(const T &val) noexcept {
		return std::isfinite(val);
	}

	/// Returns true if the input value is infinite
	/// \tparam T The type of the value
	/// \param val The value to check
	/// \return True if the value is infinite,
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isInf(const T &val) noexcept {
		return std::isinf(val);
	}

	/// Create a new number with a given magnitude and sign
	/// \tparam T The type of the magnitude
	/// \tparam M The type of the sign
	/// \param mag The magnitude of the number
	/// \param sign The value from which to copy the sign
	/// \return
	template<typename T, typename M>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T copySign(const T &mag, const M &sign) noexcept {
#if defined(LIBRAPID_MSVC)
		return std::copysign(mag, static_cast<T>(sign));
#else
		if constexpr (std::is_fundamental_v<T> && std::is_fundamental_v<M>) {
			return std::copysign(mag, static_cast<T>(sign));
		} else {
			if (sign < 0) return -mag;
			return mag;
		}
#endif
	}

	/// Extract the sign bit from a value
	/// \tparam T The type of the value
	/// \param val The value to extract the sign bit from
	/// \return The sign bit of the value
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool signBit(const T &val) noexcept {
		return signBit((double)val);
	}

	/// Extract the sign bit from a value
	/// \param val The value to extract the sign bit from
	/// \return The sign bit of the value
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool signBit(const long double &val) noexcept {
		return std::signbit(val);
	}

	/// Extract the sign bit from a value
	/// \param val The value to extract the sign bit from
	/// \return The sign bit of the value
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool signBit(const double &val) noexcept {
		return std::signbit(val);
	}

	/// Extract the sign bit from a value
	/// \param val The value to extract the sign bit from
	/// \return The sign bit of the value
	template<>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool signBit(const float &val) noexcept {
		return std::signbit(val);
	}

	/// Return a value multiplied by 2 raised to the power of an exponent
	/// \tparam T The type of the value
	/// \param x The value to multiply
	/// \param exp The exponent to raise 2 to
	/// \return x * 2^exp
	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T ldexp(const T &x, const int64_t exp) noexcept {
		return std::ldexp(x, (int)exp);
	}
} // namespace librapid

#endif // LIBRAPID_UTILS_MEMUTILS_HPP