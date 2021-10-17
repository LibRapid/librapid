#ifndef NDARRAY_CORE_MATH
#define NDARRAY_CORE_MATH

#include <librapid/config.hpp>

#include <vector>
#include <cmath>

namespace librapid
{
	constexpr long double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286;
	constexpr long double twopi = 6.283185307179586476925286766559005768394338798750211641949889184615632812572;
	constexpr long double halfpi = 1.570796326794896619231321691639751442098584699687552910487472296153908203143;
	constexpr long double e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353;
	constexpr long double sqrt2 = 1.414213562373095048801688724209698078569671875376948073176679737990732478;
	constexpr long double sqrt3 = 1.7320508075688772935274463415058723669428052538103806280558069794519330169;
	constexpr long double sqrt5 = 2.2360679774997896964091736687312762354406183596115257242708972454105209256378;

	int64_t product(const std::vector<int64_t> &vals);
	int64_t product(const int64_t *vals, int64_t num);
	double product(const std::vector<double> &vals);
	double product(const double *vals, int64_t num);

	bool anyBelow(const std::vector<int64_t> &vals, int64_t bound);
	bool anyBelow(const int64_t *vals, int64_t dims, int64_t bound);

	template<typename T>
	T min(const std::vector<T> &vals)
	{
		T min_found = 0;
		for (const auto &val : vals)
			if (val < min_found)
				min_found = val;
		return min_found;
	}

	template<typename T>
	T &&min(T &&val)
	{
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	inline auto min(T0 &&val1, T1 &&val2, Ts &&... vs)
	{
		return (val1 < val2) ?
			min(val1, std::forward<Ts>(vs)...) :
			min(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	T max(const std::vector<T> &vals)
	{
		T min_found = 0;
		for (const auto &val : vals)
			if (val > min_found)
				min_found = val;
		return min_found;
	}

	template<typename T>
	inline T &&max(T &&val)
	{
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	inline auto max(T0 &&val1, T1 &&val2, Ts &&... vs)
	{
		return (val1 > val2) ?
			max(val1, std::forward<Ts>(vs)...) :
			max(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	inline T abs(T a)
	{
		return std::abs(a);
	}

	double map(double val,
			   double start1, double stop1,
			   double start2, double stop2);

	double random(double lower, double upper);
	int64_t randint(int64_t lower, int64_t upper);

	double pow10(int64_t exponent);

	double round(const double num, int64_t dp = 0);
	double roundSigFig(const double num, int64_t figs = 3);
}

#endif // NDARRAY_CORE_MATH