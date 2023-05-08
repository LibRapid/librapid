#ifndef LIBRAPID_MATH_UTLIITY_FUNCTIONS_HPP
#define LIBRAPID_MATH_UTLIITY_FUNCTIONS_HPP

namespace librapid {
	/// \brief Limit a value to a specified range
	///
	/// \f$ C(x, m, M) = \left\{ \begin{align*} x & \quad m \le x \le M \\ m & \quad x < m \\ M &
	/// \quad x > M \end{align*}\right. \f$
	///
	/// If M < m, the values are swapped to make the function valid.
	/// For example, `clamp(5, 10, 0)` still returns `5`.
	///
	/// \tparam X Type of \p x
	/// \tparam Lower Type of \p lowerLimit
	/// \tparam Upper Type of \p upperLimit
	/// \param x Value to limit
	/// \param lowerLimit Lower bound (m)
	/// \param upperLimit Upper bound (M)
	/// \return \p x limited to the range [\p lowerLimit, \p upperLimit]
	template<typename X, typename Lower, typename Upper,
			 typename std::enable_if_t<
			   typetraits::TypeInfo<X>::type == detail::LibRapidType::Scalar &&
				 typetraits::TypeInfo<Lower>::type == detail::LibRapidType::Scalar &&
				 typetraits::TypeInfo<Upper>::type == detail::LibRapidType::Scalar,
			   int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE X clamp(X x, Lower lowerLimit, Upper upperLimit) {
		LIBRAPID_ASSERT(lowerLimit < upperLimit, "Lower limit must be below upper limit");
		if (x < lowerLimit) return static_cast<X>(lowerLimit);
		if (x > upperLimit) return static_cast<X>(upperLimit);
		return x;
	}

	/// \brief Linearly interpolate between two values
	///
	/// \f$ \mathrm{lerp}(t, L, U) = L+t\left( U-L \right) \f$
	///
	/// \tparam T Type of \p t
	/// \tparam Lower Type of \p lower
	/// \tparam Upper Type of \p upper
	/// \param t Interpolation Percentage
	/// \param lower Lower bound (L)
	/// \param upper Upper bound (U)
	/// \return
	template<typename T, typename Lower, typename Upper,
			 typename std::enable_if_t<
			   typetraits::TypeInfo<T>::type == detail::LibRapidType::Scalar &&
				 typetraits::TypeInfo<Lower>::type == detail::LibRapidType::Scalar &&
				 typetraits::TypeInfo<Upper>::type == detail::LibRapidType::Scalar,
			   int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T lerp(T t, Lower lower, Upper upper) {
		return static_cast<T>(lower) + (static_cast<T>(upper) - static_cast<T>(lower)) * t;
	}

	/// \brief Smoothly interpolate between two values
	///
	/// This smooth step implementation is based on Ken Perlin's algorithm.
	/// \f$ S(x)= \begin{cases}0 & x \leq 0 \\ 6 x^5-15 x^4+10 x^3 & 0 \leq x \leq 1 \\ 1 & 1 \leq
	/// x\end{cases} \f$
	///
	/// This function allows you to specify a lower and upper edge, which can be used to scale
	/// the range of inputs.
	///
	/// \tparam T Type of \p t
	/// \tparam Lower Type of \p lowerEdge
	/// \tparam Upper Type of \p upperEdge
	/// \param t Value to smooth step
	/// \param lowerEdge At t=lowerEdge, the function returns 0
	/// \param upperEdge At t=upperEdge, the function returns 1
	/// \return \p t interpolated between \p lowerEdge and \p upperEdge
	template<typename T, typename Lower = T, typename Upper = T,
			 typename std::enable_if_t<
			   typetraits::TypeInfo<T>::type == detail::LibRapidType::Scalar &&
				 typetraits::TypeInfo<Lower>::type == detail::LibRapidType::Scalar &&
				 typetraits::TypeInfo<Upper>::type == detail::LibRapidType::Scalar,
			   int> = 0>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T smoothStep(T t, Lower lowerEdge = 0,
														   Upper upperEdge = 1) {
		T tt = clamp((t - lowerEdge) / (upperEdge - lowerEdge), 0.0, 1.0);
		return tt * tt * tt * (tt * (tt * T(6) - T(15)) + T(10));
	}

	/// \brief Returns true if the two values are within the given tolerance of each other
	///
	/// This function is often used to compare floating point values for equality, since floating
	/// point rounding errors can cause exact equality checks to fail.
	///
	/// \tparam V1 Data type of the first value
	/// \tparam V2 Data type of the second value
	/// \tparam T Data type of the tolerance value
	/// \tparam T Data type of the tolerance value
	/// \param val1 First value
	/// \param val2 Second value
	/// \param tolerance Tolerance
	/// \return True if values are close
	template<typename V1, typename V2, typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE bool isClose(const V1 &val1, const V2 &val2,
														   const T &tolerance = 1e-6) {
		return ::librapid::abs(val1 - val2) < tolerance;
	}
} // namespace librapid

#endif // LIBRAPID_MATH_UTLIITY_FUNCTIONS_HPP