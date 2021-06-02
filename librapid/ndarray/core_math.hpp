#ifndef NDARRAY_CORE_MATH
#define NDARRAY_CORE_MATH

#include <cmath>
#include <vector>

namespace librapid
{
	namespace ndarray
	{
		namespace math
		{
			template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
			ND_INLINE const T product(const std::vector<T> &vals)
			{
				T res = 1;
				for (const auto &val : vals)
					res *= val;
				return res;
			}

			template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
			ND_INLINE const T product(const T *__restrict vals, nd_int num)
			{
				T res = 1;
				for (nd_int i = 0; i < num; i++)
					res *= vals[i];
				return res;
			}

			template<typename T, typename V>
			ND_INLINE const bool anyBelow(const std::vector<T> &vals, V bound)
			{
				for (const auto &val : vals)
					if (val < bound)
						return true;
				return false;
			}

			template<typename T, typename V>
			ND_INLINE const bool anyBelow(const T *__restrict vals, nd_int dims, V bound)
			{
				for (nd_int i = 0; i < dims; i++)
					if (vals[i] < bound)
						return true;
				return false;
			}

			template<typename T, typename V>
			ND_INLINE const T nd_to_scalar(const std::vector<T> &index, const std::vector<V> &shape)
			{
				T sig = 1, pos = 0;

				for (T i = shape.size(); i > 0; i--)
				{
					pos += (i - 1 < index.size() ? index[i - 1] : 0) * sig;
					sig *= shape[i - 1];
				}

				return pos;
			}

			template<typename A, typename B>
			ND_INLINE const A max_value(A a, B b)
			{
				if (a > b) return a;
				return b;
			}

			template<typename A, typename B>
			ND_INLINE const A min_value(A a, B b)
			{
				if (a < b) return a;
				return b;
			}

			template<typename T>
			inline T &&min(T &&val)
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
		}
	}
}

#endif // NDARRAY_CORE_MATH