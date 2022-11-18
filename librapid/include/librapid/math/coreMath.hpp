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
} // namespace librapid

#endif // LIBRAPID_MATH_CORE_MATH_HPP
