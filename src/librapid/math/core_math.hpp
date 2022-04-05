#ifndef LIBRAPID_CORE_MATH
#define LIBRAPID_CORE_MATH

#include <cmath>
#include <librapid/autocast/custom_complex.hpp>
#include <librapid/config.hpp>
#include <librapid/utils/time_utils.hpp>
#include <random>
#include <vector>

namespace librapid {
	int64_t product(const std::vector<int64_t> &vals);
	int64_t product(const int64_t *vals, int64_t num);

	double product(const std::vector<double> &vals);
	double product(const double *vals, int64_t num);

	bool anyBelow(const std::vector<int64_t> &vals, int64_t bound);
	bool anyBelow(const int64_t *vals, int64_t dims, int64_t bound);

	template<typename T>
	T min(const std::vector<T> &vals) {
		T min_found = 0;
		for (const auto &val : vals)
			if (val < min_found) min_found = val;
		return min_found;
	}

	template<typename T>
	T &&min(T &&val) {
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	inline auto min(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 < val2) ? min(val1, std::forward<Ts>(vs)...)
							 : min(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	T max(const std::vector<T> &vals) {
		T min_found = 0;
		for (const auto &val : vals)
			if (val > min_found) min_found = val;
		return min_found;
	}

	template<typename T>
	inline T &&max(T &&val) {
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	inline auto max(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 > val2) ? max(val1, std::forward<Ts>(vs)...)
							 : max(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	inline T abs(T a) {
		return std::abs(a);
	}

	template<typename A, typename B,
			 typename std::enable_if_t<std::is_fundamental_v<A> &&
									   std::is_fundamental_v<B>> = 0>
	inline A pow(A a, B exp) {
		return std::pow(a, exp);
	}

	template<typename T>
	inline T sqrt(T a) {
		return std::sqrt(a);
	}

	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>> = 0>
	inline T exp(T a) {
		return std::exp(a);
	}

	template<typename A, typename B>
	inline Complex<A> pow(Complex<A> a, B exp) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	template<typename A, typename B>
	inline Complex<A> pow(A a, const Complex<B> &exp) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	template<typename A, typename B>
	inline Complex<A> pow(const Complex<A> &a, const Complex<B> &exp) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	template<typename T>
	inline Complex<T> sqrt(Complex<T> a) {
		throw std::runtime_error(
		  "Complex number square roots have not yet been implemented");
	}

	template<typename T>
	inline Complex<T> exp(Complex<T> a) {
		throw std::runtime_error(
		  "Complex number exponents have not yet been implemented");
	}

	// Just some helpful utility functions
	inline vcl::Vec8d pow(const vcl::Vec8d &a, const vcl::Vec8d &power) {
		return vcl::Vec8d(std::pow(a[0], power[0]),
						  std::pow(a[1], power[1]),
						  std::pow(a[2], power[2]),
						  std::pow(a[3], power[3]),
						  std::pow(a[4], power[4]),
						  std::pow(a[5], power[5]),
						  std::pow(a[6], power[6]),
						  std::pow(a[7], power[7]));
	}

	inline vcl::Vec16f pow(const vcl::Vec16f &a, const vcl::Vec16f &power) {
		return vcl::Vec16f(std::pow(a[0], power[0]),
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
						   std::pow(a[15], power[15]));
	}

	template<typename T>
	inline T sin(T a) {
		return std::sin(a);
	}

	template<typename T>
	inline T cos(T a) {
		return std::cos(a);
	}

	template<typename T>
	inline T tan(T a) {
		return std::tan(a);
	}

	template<typename T>
	inline T asin(T a) {
		return std::asin(a);
	}

	template<typename T>
	inline T acos(T a) {
		return std::acos(a);
	}

	template<typename T>
	inline T atan(T a) {
		return std::atan(a);
	}

	template<typename T>
	inline T sinh(T a) {
		return std::sinh(a);
	}

	template<typename T>
	inline T cosh(T a) {
		return std::cosh(a);
	}

	template<typename T>
	inline T tanh(T a) {
		return std::tanh(a);
	}

	template<typename T>
	inline T asinh(T a) {
		return std::asinh(a);
	}

	template<typename T>
	inline T acosh(T a) {
		return std::acosh(a);
	}

	template<typename T>
	inline T atanh(T a) {
		return std::atanh(a);
	}

	template<typename T>
	inline Complex<T> sin(const Complex<T> &a) {
		return {sin(a.real()) * cosh(a.imag()), cos(a.real()) * sinh(a.imag())};
	}

	template<typename T>
	inline Complex<T> cos(const Complex<T> &a) {
		return {cos(a.real()) * cosh(a.imag()), sin(a.real()) * sinh(a.imag())};
	}

	template<typename T>
	inline Complex<T> tan(const Complex<T> &a) {
		return sin(a) / cos(a);
	}

	template<typename T>
	inline Complex<T> &asin(const Complex<T> &a) {
		throw std::runtime_error(
		  "Inverse trigonometric functions of complex values have not yet been "
		  "implemented");
	}

	template<typename T>
	inline Complex<T> &acos(const Complex<T> &a) {
		throw std::runtime_error(
		  "Inverse trigonometric functions of complex values have not yet been "
		  "implemented");
	}

	template<typename T>
	inline Complex<T> &atan(const Complex<T> &a) {
		throw std::runtime_error(
		  "Inverse trigonometric functions of complex values have not yet been "
		  "implemented");
	}

	template<typename T>
	inline Complex<T> sinh(const Complex<T> &a) {
		return {sinh(a.real()) * cos(a.imag()), cosh(a.real()) * sin(a.imag())};
	}

	template<typename T>
	inline Complex<T> cosh(const Complex<T> &a) {
		return {cosh(a.real()) * cos(a.imag()), sinh(a.real()) * sin(a.imag())};
	}

	template<typename T>
	inline Complex<T> tanh(const Complex<T> &a) {
		return sinh(a) / cosh(a);
	}

	template<typename T>
	inline Complex<T> &asinh(const Complex<T> &a) {
		throw std::runtime_error(
		  "Inverse trigonometric functions of complex values have not yet been "
		  "implemented");
	}

	template<typename T>
	inline Complex<T> &acosh(const Complex<T> &a) {
		throw std::runtime_error(
		  "Inverse trigonometric functions of complex values have not yet been "
		  "implemented");
	}

	template<typename T>
	inline Complex<T> &atanh(const Complex<T> &a) {
		throw std::runtime_error(
		  "Inverse trigonometric functions of complex values have not yet been "
		  "implemented");
	}

	[[nodiscard]] double map(double val, double start1, double stop1,
							 double start2, double stop2);

	template<typename T = double>
	inline T random(T lower = 0, T upper = 1, uint64_t seed = -1) {
		// Random floating point value in range [lower, upper)

		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator(
		  seed == (uint64_t)-1 ? (unsigned int)(seconds() * 10) : seed);
		return (T)(lower + (upper - lower) * distribution(generator));
	}

	template<typename T>
	inline Complex<T> random(const Complex<T> &min, const Complex<T> &max,
							 uint64_t seed = -1) {
		return {random<T>(min.real(), max.real(), seed),
				random<T>(min.imag(), max.imag(), seed)};
	}

	inline int64_t randint(int64_t lower, int64_t upper, uint64_t seed = -1) {
		// Random integral value in range [lower, upper]
		return (int64_t)random(
		  (double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1, seed);
	}

	inline double trueRandomEntropy() {
		static std::random_device rd;
		return rd.entropy();
	}

	template<typename T = double>
	inline double trueRandom(T lower = 0, T upper = 1) {
		// Truly random value in range [lower, upper)
		static std::random_device rd;
		std::uniform_real_distribution<double> dist((double)lower,
													(double)upper);
		return dist(rd);
	}

	inline int64_t trueRandint(int64_t lower, int64_t upper) {
		// Truly random value in range [lower, upper)
		return (int64_t)trueRandom((double)(lower - (lower < 0 ? 1 : 0)),
								   (double)upper + 1);
	}

	/**
	 * Adapted from
	 * https://docs.oracle.com/javase/6/docs/api/java/util/Random.html#nextGaussian()
	 */
	inline double randomGaussian() {
		static double nextGaussian;
		static bool hasNextGaussian = false;

		double res;
		if (hasNextGaussian) {
			hasNextGaussian = false;
			res				= nextGaussian;
		} else {
			double v1, v2, s;
			do {
				v1 = 2 * random() - 1; // between -1.0 and 1.0
				v2 = 2 * random() - 1; // between -1.0 and 1.0
				s  = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);
			double multiplier = sqrt(-2 * log(s) / s);
			nextGaussian	  = v2 * multiplier;
			hasNextGaussian	  = true;
			res				  = v1 * multiplier;
		}

		return res;
	}

	[[nodiscard]] double pow10(int64_t exponent);

	[[nodiscard]] double round(double num, int64_t dp = 0);

	[[nodiscard]] double roundSigFig(double num, int64_t figs = 3);

	uint64_t nthFibonacci(uint8_t n);

	void betterFcknBeEven(int64_t n);
} // namespace librapid

#endif // LIBRAPID_CORE_MATH