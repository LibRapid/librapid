#ifndef LIBRAPID_PATCH_MATH
#define LIBRAPID_PATCH_MATH

#include <cmath>
#include <librapid/autocast/custom_complex.hpp>
#include <librapid/config.hpp>
#include <librapid/utils/time_utils.hpp>
#include <random>
#include <vector>

namespace librapid {
	template<typename A, typename B>
	inline A pow_numeric_only(A a, B exp) {
		return std::pow(a, exp);
	}

	template<typename A, typename B>
	inline Complex<A> pow_numeric_only(Complex<A> a, B exp) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	template<typename A, typename B>
	inline Complex<A> pow_numeric_only(A a, const Complex<B> &exp) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	template<typename A, typename B>
	inline Complex<A> pow_numeric_only(const Complex<A> &a, const Complex<B> &exp) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	inline vcl::Vec8d pow_numeric_only(const vcl::Vec8d &a,
									   const vcl::Vec8d &power) {
		return {std::pow(a[0], power[0]),
				std::pow(a[1], power[1]),
				std::pow(a[2], power[2]),
				std::pow(a[3], power[3]),
				std::pow(a[4], power[4]),
				std::pow(a[5], power[5]),
				std::pow(a[6], power[6]),
				std::pow(a[7], power[7])};
	}

	inline vcl::Vec16f pow_numeric_only(const vcl::Vec16f &a,
										const vcl::Vec16f &power) {
		return {std::pow(a[0], power[0]),
				std::pow(a[1], power[1]),
				std::pow(a[2], power[2]),
				std::pow(a[3], power[3]),
				std::pow(a[4], power[4]),
				std::pow(a[5], power[5]),
				std::pow(a[6], power[6]),
				std::pow(a[7], power[7]),
				std::pow(a[8], power[8]),
				std::pow(a[9], power[9]),
				std::pow(a[10], power[10]),
				std::pow(a[11], power[11]),
				std::pow(a[12], power[12]),
				std::pow(a[13], power[13]),
				std::pow(a[14], power[14]),
				std::pow(a[15], power[15])};
	}
} // namespace librapid

#endif // LIBRAPID_PATCH_MATH