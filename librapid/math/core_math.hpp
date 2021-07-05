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
		constexpr long double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286;
		constexpr long double twopi = 6.283185307179586476925286766559005768394338798750211641949889184615632812572;
		constexpr long double halfpi = 1.570796326794896619231321691639751442098584699687552910487472296153908203143;
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
		LR_INLINE T product(const T *__restrict vals, lr_int num)
		{
			T res = 1;
			for (lr_int i = 0; i < num; i++)
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
		LR_INLINE const bool anyBelow(const T *__restrict vals, lr_int dims, V bound)
		{
			for (lr_int i = 0; i < dims; i++)
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

		template<typename T>
		LR_INLINE T &&min_max_forward(T &&val)
		{
			return std::forward<T>(val);
		}

		template<typename T0, typename T1, typename... Ts>
		LR_INLINE auto min(T0 &&val1, T1 &&val2, Ts &&... vs)
		{
			return (val1 < val2) ?
				min_max_forward(val1, std::forward<Ts>(vs)...) :
				min_max_forward(val2, std::forward<Ts>(vs)...);
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

		template<typename T0, typename T1, typename... Ts>
		LR_INLINE auto max(T0 &&val1, T1 &&val2, Ts &&... vs)
		{
			return (val1 > val2) ?
				min_max_forward(val1, std::forward<Ts>(vs)...) :
				min_max_forward(val2, std::forward<Ts>(vs)...);
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

		template<typename T, typename std::enable_if<std::is_signed<T>::value, int>::type = 0>
		LR_INLINE T abs(T a)
		{
			if (a < 0)
				return -a;
			return a;
		}

		template<typename T, typename std::enable_if<std::is_unsigned<T>::value, int>::type = 0>
		LR_INLINE T abs(T a)
		{
			return a;
		}

		template<typename v, typename s, typename e, typename ss, typename ee>
		LR_INLINE typename std::common_type<v, s, e, ss, ee>::type map(v val, s start1, e stop1,
																	   ss start2, ee stop2)
		{
			using _Ty = typename std::common_type<v, s, e, ss, ee>::type;
			return (_Ty) start2 + ((_Ty) stop2 - (_Ty) start2) *
				(((_Ty) val - (_Ty) start1) / ((_Ty) stop1 - (_Ty) start1));
		}

		template<typename type, typename std::enable_if<std::is_floating_point<type>::value, int>::type = 0>
		LR_INLINE type random(const type &min, const type &max)
		{
			// Random floating point value in range [min, max)

			static std::uniform_real_distribution<type> distribution(0., 1.);
			static std::mt19937 generator((unsigned int) (TIME * 10));
			return min + (max - min) * distribution(generator);
		}

		template<typename type, typename std::enable_if<std::is_integral<type>::value, int>::type = 0>
		LR_INLINE type random(const type &min, const type &max)
		{
			// Random integral value in range [min, max]
			return (type) random((double) min, (double) max + 1);
		}

		LR_INLINE double pow10(lr_int exponent)
		{
			const static double pows[] = {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
			if (exponent >= -5 && exponent <= 5)
				return pows[exponent + 5];

			double res = 1;

			if (exponent > 0)
				for (lr_int i = 0; i < exponent; i++)
					res *= 10;
			else
				for (lr_int i = 0; i > exponent; i--)
					res *= 0.1;

			return res;
		}

		template<typename T>
		LR_INLINE T round(const T num, lr_int dp = 0)
		{
			static double alpha = std::numeric_limits<T>::digits10 > 10 ? 0.4999999999 : 0.49999;

			double p10 = pow10(-dp);

			double remainder = fmod((double) num, p10);

			if (remainder == 0)
				return num;

			if (remainder < alpha * p10)
				return (T) (num - remainder);

			return (T) (num + p10 - remainder);
		}

		template<typename T>
		LR_INLINE T round_sigfig(const T num, lr_int figs = 3)
		{
			if (figs <= 0)
				throw std::invalid_argument("Cannot round to "
											+ std::to_string(figs)
											+ " significant figures. Must be greater than 0");

			T tmp = num > 0 ? num : -num;
			lr_int n = 0;

			while (tmp > 10)
			{
				tmp /= 10;
				++n;
			}

			while (tmp < 1)
			{
				tmp *= 10;
				--n;
			}

			return (tmp > 0 ? 1 : -1) * (round(tmp, figs - 1) * pow10(n));
		}
	}
}

#endif // NDARRAY_CORE_MATH