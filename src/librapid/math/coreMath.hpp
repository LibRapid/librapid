#ifndef LIBRAPID_CORE_MATH
#define LIBRAPID_CORE_MATH

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "../array/traits.hpp"

namespace librapid {
	LR_INLINE int64_t product(const std::vector<int64_t> &vals) {
		int64_t res = 1;
		for (const auto &val : vals) res *= val;
		return res;
	}

	LR_INLINE double product(const std::vector<double> &vals) {
		double res = 1;
		for (const auto &val : vals) res *= val;
		return res;
	}

	template<typename T>
	LR_INLINE T min(const std::vector<T> &vals) {
		T min_found = 0;
		for (const auto &val : vals)
			if (val < min_found) min_found = val;
		return min_found;
	}

	template<typename T>
	LR_INLINE T &&min(T &&val) {
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	LR_INLINE auto min(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 < val2) ? min(val1, std::forward<Ts>(vs)...)
							 : min(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	LR_INLINE T max(const std::vector<T> &vals) {
		T min_found = 0;
		for (const auto &val : vals)
			if (val > min_found) min_found = val;
		return min_found;
	}

	template<typename T>
	LR_INLINE T &&max(T &&val) {
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	LR_INLINE auto max(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 > val2) ? max(val1, std::forward<Ts>(vs)...)
							 : max(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	LR_INLINE T abs(T a) {
		return std::abs(a);
	}

	template<typename A, typename B,
			 typename std::enable_if_t<std::is_fundamental_v<A> && std::is_fundamental_v<B>> = 0>
	LR_INLINE A pow(A a, B exp) {
		return std::pow(a, exp);
	}

	template<typename T>
	LR_INLINE T sqrt(T a) {
		return std::sqrt(a);
	}

	template<typename T>
	LR_INLINE T exp(T a) {
		return std::exp(a);
	}

	template<typename T>
	LR_INLINE T pow(T a, T power) {
		return std::pow(a, power);
	}

	template<typename T>
	LR_INLINE T ln(T a) {
		return std::log(a);
	}

	template<typename T>
	LR_INLINE T log2(T a) {
		return std::log2(a);
	}

	template<typename T>
	LR_INLINE T log10(T a) {
		return std::log10(a);
	}

	template<typename T>
	LR_INLINE T log(T a, T base) {
		return ln(a) / ln(base);
	}

	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>> = 0>
	LR_INLINE T exp(T a) {
		return std::exp(a);
	}

	template<typename T>
	LR_INLINE T sin(T a) {
		return std::sin(a);
	}

	template<typename T>
	LR_INLINE T cos(T a) {
		return std::cos(a);
	}

	template<typename T>
	LR_INLINE T tan(T a) {
		return std::tan(a);
	}

	template<typename T>
	LR_INLINE T asin(T a) {
		return std::asin(a);
	}

	template<typename T>
	LR_INLINE T acos(T a) {
		return std::acos(a);
	}

	template<typename T>
	LR_INLINE T atan(T a) {
		return std::atan(a);
	}

	template<typename T>
	LR_INLINE T sinh(T a) {
		return std::sinh(a);
	}

	template<typename T>
	LR_INLINE T cosh(T a) {
		return std::cosh(a);
	}

	template<typename T>
	LR_INLINE T tanh(T a) {
		return std::tanh(a);
	}

	template<typename T>
	LR_INLINE T asinh(T a) {
		return std::asinh(a);
	}

	template<typename T>
	LR_INLINE T acosh(T a) {
		return std::acosh(a);
	}

	template<typename T>
	LR_INLINE T atanh(T a) {
		return std::atanh(a);
	}

	template<typename T>
	LR_INLINE T map(T val, T start1, T stop1, T start2, T stop2) {
		return start2 + (stop2 - start2) * ((val - start1) / (stop1 - start1));
	}

	LR_INLINE double random(double lower = 0, double upper = 1, uint64_t seed = -1) {
		// Random floating point value in range [lower, upper)
		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator(seed == (uint64_t)-1 ? (unsigned int)(now() * 10) : seed);
		return lower + (upper - lower) * distribution(generator);
	}

	template<typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
	LR_INLINE T randint(T lower, T upper, uint64_t seed = -1) {
		// Random integral value in range [lower, upper]
		return (int64_t)random((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1, seed);
	}

	LR_INLINE double trueRandomEntropy() {
		static std::random_device rd;
		return rd.entropy();
	}

	template<typename T = double>
	LR_INLINE double trueRandom(T lower = 0, T upper = 1) {
		// Truly random value in range [lower, upper)
		static std::random_device rd;
		std::uniform_real_distribution<double> dist((double)lower, (double)upper);
		return dist(rd);
	}

	LR_INLINE int64_t trueRandint(int64_t lower, int64_t upper) {
		// Truly random value in range [lower, upper)
		return (int64_t)trueRandom((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1);
	}

	/**
	 * Adapted from
	 * https://docs.oracle.com/javase/6/docs/api/java/util/Random.html#nextGaussian()
	 */
	LR_INLINE double randomGaussian() {
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
			double multiplier = sqrt(-2 * ln(s) / s);
			nextGaussian	  = v2 * multiplier;
			hasNextGaussian	  = true;
			res				  = v1 * multiplier;
		}

		return res;
	}

	template<typename T = double>
	LR_INLINE auto pow10(int64_t exponent) {
		using Scalar = typename internal::traits<T>::Scalar;

		const static Scalar pows[] = {
		  0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
		if (exponent >= -5 && exponent <= 5) return pows[exponent + 5];

		Scalar res = 1;

		if (exponent > 0)
			for (int64_t i = 0; i < exponent; i++) res *= 10;
		else
			for (int64_t i = 0; i > exponent; i--) res *= 0.1;

		return res;
	}

	template<typename T1, typename T2,
			 typename std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>, int> = 0>
	LR_INLINE auto mod(T1 val, T2 divisor) {
		return val % divisor;
	}

	template<typename T1, typename T2,
			 typename std::enable_if_t<std::is_floating_point_v<T1> || std::is_floating_point_v<T2>,
									   int> = 0>
	LR_INLINE auto mod(T1 val, T2 divisor) {
		return std::fmod(val, divisor);
	}

	namespace roundMode {
		// Rounding Mode Information:
		// Bit mask:
		// [0] -> Round up if difference >= 0.5
		// [1] -> Round up if difference < 0.5
		// [2] -> Round to nearest even
		// [3] -> Round to nearest odd
		// [4] -> Round only if difference == 0.5

		static constexpr int8_t UP		  = 0b00000011;
		static constexpr int8_t DOWN	  = 0b00000000;
		static constexpr int8_t TRUNC	  = 0b00000000;
		static constexpr int8_t HALF_EVEN = 0b00010100;
		static constexpr int8_t MATH	  = 0b00000001;
	} // namespace roundMode

	template<typename T = double>
	LR_INLINE auto round(T num, int64_t dp, int8_t mode = roundMode::MATH) {
		using Scalar = typename internal::traits<T>::Scalar;

		const Scalar alpha	= pow10<T>(dp);
		const Scalar beta	= pow10<T>(-dp);
		const Scalar absNum = abs(num * alpha);
		Scalar y			= floor(absNum);
		Scalar diff			= absNum - y;
		if (mode & (1 << 0) && diff >= 0.5) y += 1;
		if (mode & (1 << 2)) {
			auto integer	 = (uint64_t)y;
			auto nearestEven = (integer & 1) ? (y + 1) : (Scalar)integer;
			if (mode & (1 << 4) && diff == 0.5) y = nearestEven;
		}
		return (num >= 0 ? y : -y) * beta;
	}

	template<typename T1 = double, typename T2 = double>
	LR_INLINE typename std::common_type_t<T1, T2> roundTo(T1 num, T2 val) {
		auto rem = mod(num, val);
		if (rem >= val / 2) return (num + val) - rem;
		return num - rem;
	}

	template<typename T1 = double, typename T2 = double>
	LR_INLINE typename std::common_type_t<T1, T2> roundUpTo(T1 num, T2 val) {
		auto rem = mod(num, val);
		if (rem == 0) return num;
		return (num + val) - rem;
	}

	template<typename T>
	LR_INLINE T roundSigFig(T num, int64_t figs) {
		LR_ASSERT(figs > 0,
				  "Cannot round to {} significant figures. Value must be greater than zero",
				  figs);

		T tmp	  = num > 0 ? num : -num;
		int64_t n = 0;

		while (tmp > 10) {
			tmp /= 10;
			++n;
		}

		while (tmp < 1) {
			tmp *= 10;
			--n;
		}

		return (tmp > 0 ? 1 : -1) * (round(tmp, figs - 1) * pow10<T>(n));
	}

	template<typename T, typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
	LR_INLINE T lerp(T _a, T _b, T _t) {
		if (std::isnan(_a) || std::isnan(_b) || std::isnan(_t))
			return std::numeric_limits<T>::quiet_NaN();
		else if ((_a <= T {0} && _b >= T {0}) || (_a >= T {0} && _b <= T {0}))
		// ab <= 0 but product could overflow.
#ifndef FMA
			return _t * _b + (T {1} - _t) * _a;
#else
			return std::fma(_t, __b, (_Float {1} - _t) * __a);
#endif
		else if (_t == T {1})
			return _b;
		else { // monotonic near t == 1.
#ifndef FMA
			const auto _x = _a + _t * (_b - _a);
#else
			const auto _x = std::fma(_t, __b - __a, __a);
#endif
			return (_t > T {1}) == (_b > _a) ? max(_b, _x) : min(_b, _x);
		}
	}
} // namespace librapid

#endif // LIBRAPID_CORE_MATH