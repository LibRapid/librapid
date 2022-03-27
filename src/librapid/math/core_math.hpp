#ifndef LIBRAPID_CORE_MATH
#define LIBRAPID_CORE_MATH

#include <cmath>
#include <librapid/autocast/custom_complex.hpp>
#include <librapid/config.hpp>
#include <librapid/utils/time_utils.hpp>
#include <random>
#include <vector>

namespace librapid {
	//	[[maybe_unused]] constexpr long double pi
	//= 3.141592653589793238462643383279502884197169399375105820974944592307816406286;
	//	[[maybe_unused]] constexpr long double twopi
	//= 6.283185307179586476925286766559005768394338798750211641949889184615632812572;
	//	[[maybe_unused]] constexpr long double halfpi
	//= 1.570796326794896619231321691639751442098584699687552910487472296153908203143;
	//	[[maybe_unused]] constexpr long double e
	//= 2.718281828459045235360287471352662497757247093699959574966967627724076630353;
	//	[[maybe_unused]] constexpr long double sqrt2
	//= 1.414213562373095048801688724209698078569671875376948073176679737990732478;
	//	[[maybe_unused]] constexpr long double sqrt3
	//= 1.7320508075688772935274463415058723669428052538103806280558069794519330169;
	//	[[maybe_unused]] constexpr long double sqrt5
	//= 2.2360679774997896964091736687312762354406183596115257242708972454105209256378;

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

	[[nodiscard]] double map(double val, double start1, double stop1,
							 double start2, double stop2);

	template<typename T = double>
	inline T random(T lower = 0, T upper = 1, uint64_t seed = -1) {
		// Random floating point value in range [lower, upper)

		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator(
		  seed == (uint64_t)-1 ? (unsigned int)(seconds() * 10) : seed);
		return (T) (lower + (upper - lower) * distribution(generator));
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
		// Truly random value in range [lower, upper)
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