#ifndef NDARRAY_UTILS
#define NDARRAY_UTILS

#include <cstring>
#include <librapid/math/rapid_math.hpp>

namespace librapid
{
	namespace utils
	{
		template<typename T>
		inline std::vector<T> sub_vector(const std::vector<T> &vec,
										 lr_int start = (lr_int) -1,
										 lr_int end = (lr_int) -1)
		{
			auto s = vec.begin();
			auto e = vec.end();

			if (start != (lr_int) -1) s += start;
			if (end != (lr_int) -1) e -= end;

			return std::vector<T>(s, e);
		}

		template<typename T>
		inline auto sub_vector(const T *vec, lr_int len,
							   lr_int start = (lr_int) -1,
							   lr_int end = (lr_int) -1)
		{
			lr_int s = start == (lr_int) -1 ? 0 : start;
			lr_int e = end == (lr_int) -1 ? len : end;
			auto res = new T[e - s];
			memcpy(res, vec + s, sizeof(T) * (e - s));
			return std::make_pair(res, e - s);
		}

		template<typename A, typename B>
		LR_INLINE bool check_ptr_match(A *a, lr_int len_a, B *b,
									   lr_int len_b)
		{
			if (len_a != len_b)
				return false;
			for (lr_int i = 0; i < len_a; i++)
				if (a[i] != b[i])
					return false;
			return true;
		}

		template<typename A, typename B, typename C>
		LR_INLINE bool check_ptr_match(A *a, lr_int len_a,
									   const std::pair<B, C> &pair,
									   bool del_pair = false)
		{
			if (len_a != pair.second)
			{
				if (del_pair) delete pair.first;
				return false;
			}
			for (lr_int i = 0; i < len_a; i++)
			{
				if (a[i] != pair.first[i])
				{
					if (del_pair) delete pair.first;
					return false;
				}
			}

			if (del_pair) delete pair.first;
			return true;
		}

		template<typename A, typename B, typename C, typename D>
		LR_INLINE bool check_ptr_match(const std::pair<A, B> &pair1,
									   const std::pair<C, D> &pair2,
									   bool del_pair1 = false,
									   bool del_pair2 = false)
		{
			if (pair1.second != pair2.second)
			{
				if (del_pair1) delete pair1.first;
				if (del_pair2) delete pair2.first;
				return false;
			}
			for (lr_int i = 0; i < pair1.second; i++)
			{
				if (pair1.first[i] != pair2.first[i])
				{
					if (del_pair1) delete pair1.first;
					if (del_pair2) delete pair2.first;
					return false;
				}
			}

			if (del_pair1) delete pair1.first;
			if (del_pair2) delete pair2.first;
			return true;
		}
	}
}

#endif // NDARRAY_UTILS