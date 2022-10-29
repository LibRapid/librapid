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
	T min(const T &val);

	/// Return the smallest value of a given set of values
	/// \tparam Types Data types of the input values
	/// \param vals Input values
	/// \return The smallest element of the input values
	template<typename First, typename... Rest>
	auto min(const First &first, const Rest &...rest);
} // namespace librapid

#endif // LIBRAPID_MATH_CORE_MATH_HPP
