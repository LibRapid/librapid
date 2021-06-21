#ifndef NDARRAY_CORE_MATH
#define NDARRAY_CORE_MATH

#include <cmath>
#include <vector>
#include <random>
#include <chrono>

namespace librapid
{
	namespace math
	{
	#define TIME (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000000

		constexpr long double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286;
		constexpr long double twoPi = 6.283185307179586476925286766559005768394338798750211641949889184615632812572;
		constexpr long double halfPi = 1.570796326794896619231321691639751442098584699687552910487472296153908203143;
		constexpr long double e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353;
		constexpr long double sqrt2 = 1.414213562373095048801688724209698078569671875376948073176679737990732478;
		constexpr long double sqrt3 = 1.7320508075688772935274463415058723669428052538103806280558069794519330169;
		constexpr long double sqrt5 = 2.2360679774997896964091736687312762354406183596115257242708972454105209256378;

		template<typename T>
		LR_INLINE T product(const std::vector<T> &vals)
		{
			T res = 1;
			for (const auto &val : vals)
				res *= val;
			return res;
		}

		template<typename T>
		LR_INLINE T product(const T *__restrict vals, nd_int num)
		{
			T res = 1;
			for (nd_int i = 0; i < num; i++)
				res *= vals[i];
			return res;
		}

		template<typename T, typename V>
		LR_INLINE const bool anyBelow(const std::vector<T> &vals, V bound)
		{
			for (const auto &val : vals)
				if (val < bound)
					return true;
			return false;
		}

		template<typename T, typename V>
		LR_INLINE const bool anyBelow(const T *__restrict vals, nd_int dims, V bound)
		{
			for (nd_int i = 0; i < dims; i++)
				if (vals[i] < bound)
					return true;
			return false;
		}

		template<typename T, typename V>
		LR_INLINE const T nd_to_scalar(const std::vector<T> &index, const std::vector<V> &shape)
		{
			T sig = 1, pos = 0;

			for (T i = shape.size(); i > 0; i--)
			{
				pos += (i - 1 < index.size() ? index[i - 1] : 0) * sig;
				sig *= shape[i - 1];
			}

			return pos;
		}

		/*
		template<typename A, typename B>
		LR_INLINE const A max_value(A a, B b)
		{
			if (a > b) return a;
			return b;
		}

		template<typename A, typename B>
		LR_INLINE const A min_value(A a, B b)
		{
			if (a < b) return a;
			return b;
		}
		*/

		template<typename T>
		LR_INLINE T &&min(T &&val)
		{
			return std::forward<T>(val);
		}

		template<typename T0, typename T1, typename... Ts>
		LR_INLINE auto min(T0 &&val1, T1 &&val2, Ts &&... vs)
		{
			return (val1 < val2) ?
				min(val1, std::forward<Ts>(vs)...) :
				min(val2, std::forward<Ts>(vs)...);
		}

		template<typename T>
		LR_INLINE auto min(const std::vector<T> &vals)
		{
			T min_found = 0;
			for (const auto &val : vals)
				if (val < min_found)
					min_found = val;
			return min_found;
		}

		template<typename T>
		LR_INLINE T &&max(T &&val)
		{
			return std::forward<T>(val);
		}

		template<typename T0, typename T1, typename... Ts>
		LR_INLINE auto max(T0 &&val1, T1 &&val2, Ts &&... vs)
		{
			return (val1 > val2) ?
				max(val1, std::forward<Ts>(vs)...) :
				max(val2, std::forward<Ts>(vs)...);
		}

		template<typename T>
		LR_INLINE auto max(const std::vector<T> &vals)
		{
			T min_found = 0;
			for (const auto &val : vals)
				if (val > min_found)
					min_found = val;
			return min_found;
		}

		template<typename t>
		LR_INLINE t abs(t a)
		{
			if (a < 0)
				return -a;
			return a;
		}

		template<>
		LR_INLINE unsigned int abs(unsigned int a)
		{
			return a;
		}

		template<>
		LR_INLINE unsigned long long abs(unsigned long long a)
		{
			return a;
		}

		template<typename v, typename s, typename e, typename ss, typename ee>
		LR_INLINE typename std::common_type<v, s, e, ss, ee>::type map(v val, s start1, e stop1, ss start2, ee stop2)
		{
			using _Ty = typename std::common_type<v, s, e, ss, ee>::type;
			return (_Ty) start2 + ((_Ty) stop2 - (_Ty) start2) * (((_Ty) val - (_Ty) start1) / ((_Ty) stop1 - (_Ty) start1));
		}

		template<typename type, typename std::enable_if<std::is_floating_point<type>::value, int>::type = 0>
		LR_INLINE type random(const type &min, const type &max)
		{
			// Random floating point value in range [min, max)

			static std::uniform_real_distribution<type> distribution(0., 1.);
			static std::mt19937 generator(TIME * 1000000);
			return min + (max - min) * distribution(generator);
		}

		template<typename type, typename std::enable_if<std::is_integral<type>::value, int>::type = 0>
		LR_INLINE type random(const type &min, const type &max)
		{
			// Random integral value in range [min, max]

			static std::uniform_real_distribution<double> distribution(0., 1.);
			static std::mt19937 generator(TIME * 1000000);
			return (type) random((double) min, (double) max + 1);
		}

		template<typename t>
		LR_INLINE t round(const t &number, nd_int dp = 0)
		{
			auto num = abs(number);

			t remainder = fmod(abs(num), 1. * pow(10, -((t) dp)));
			if (remainder == 0)
				return num * (num >= 0 ? 1 : -1);

			if (remainder < 0.4999999999 * pow(10, -((t) dp)))
				return (num - remainder) * (num >= 0 ? 1 : -1);

			return (num + (1. * pow(10, -((t) dp))) - remainder) * (num >= 0 ? 1 : -1);
		}
	}
}

#endif // NDARRAY_CORE_MATH