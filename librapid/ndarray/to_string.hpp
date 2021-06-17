#ifndef NDARRAY_TO_STRING
#define NDARRAY_TO_STRING

#include <librapid/math/rapid_math.hpp>

namespace librapid
{
	namespace ndarray
	{
		namespace to_string
		{
			struct str_container
			{
				std::string str;
				nd_int decimal_point;
			};

			template<typename T>
			ND_INLINE bool inc_arr(std::vector<T> &arr, const std::vector<T> &m)
			{
				arr[arr.size() - 1]++;

				for (T i = 0; i < arr.size(); i++)
				{
					if (arr[arr.size() - i - 1] >= m[m.size() - i - 1])
					{
						if (arr.size() - i == 1)
							return false;

						arr[arr.size() - i - 2]++;
						arr[arr.size() - i - 1] = 0;
					}
				}

				return true;
			}

			template<typename T>
			ND_INLINE str_container format_numerical(const T &val)
			{
				std::stringstream stream;

				// if (std::is_floating_point<T>::value)
				// 	stream.precision((unsigned long long) std::log((double) math::max_value(val, 1)) + 10);

				stream << val;
				auto str = stream.str();
				auto last_decimal = str.find_last_of('.');

				if (std::is_floating_point<T>::value && last_decimal == std::string::npos)
				{
					stream << ".";
					last_decimal = stream.str().length() - 1;
				}

				str = stream.str();
				auto lastZero = str.find_last_of('0');

				// Value is integral
				if (last_decimal == std::string::npos)
					return {str, (nd_int) str.length() - 1};

				return {str, (nd_int) last_decimal};
			}

			ND_INLINE std::string to_string_1D(const std::vector<std::string> &adjusted, bool strip_middle)
			{
				std::string res = "[";

				for (nd_int i = 0; i < adjusted.size(); i++)
				{
					if (strip_middle && adjusted.size() > 6 && i == 3)
					{
						i = adjusted.size() - 3;
						res += "... ";
					}

					res += adjusted[i];
				}

				res[res.length() - 1] = ']';
				return res;
			}

			template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
			ND_INLINE std::string to_string(const std::vector<std::string> &adjusted, const std::vector<T> &shape,
											T depth, bool strip_middle)
			{
				if (shape.size() == 1)
					return to_string_1D(adjusted, strip_middle);

				if (shape.size() == 2)
				{
					std::string res = "[";

					T count = 0;
					for (T i = 0; i < adjusted.size(); i += shape[1])
					{
						if (strip_middle && shape[0] > 6 && i == shape[1] * 3)
						{
							i = (T) adjusted.size() - shape[1] * 3;
							res += std::string(depth, ' ') + "...\n";
							count = shape[0] - 3;
						}

						if (i != 0)
							res += std::string(depth, ' ');

						auto begin = adjusted.begin() + i;
						auto end = adjusted.begin() + i + shape[1];
						std::vector<std::string> substr(begin, end);
						res += to_string_1D(substr, strip_middle);

						if (count + 1 != shape[0])
							res += "\n";

						count++;
					}

					return res + "]";
				}
				else
				{
					std::string res = "[";
					T count = 0;
					T inc = math::product(shape) / shape[0];

					for (T i = 0; i < adjusted.size(); i += inc)
					{
						if (strip_middle && shape[0] > 6 && i == inc * 3)
						{
							i = adjusted.size() - inc * 3;
							res += std::string(depth, ' ') + "...\n\n";
							count = shape[0] - 3;
						}

						if (i != 0)
							res += std::string(depth, ' ');

						auto adjustedStart = adjusted.begin() + i;
						auto adjustedEnd = adjusted.begin() + i + inc;
						auto shapeStart = shape.begin() + 1;
						auto shapeEnd = shape.end();

						auto adjusted_substring = std::vector<std::string>(adjustedStart, adjustedEnd);
						auto sub_shape = std::vector<nd_int>(shapeStart, shapeEnd);

						res += to_string(adjusted_substring, sub_shape, depth + 1, strip_middle);

						if (count + 1 != shape[0])
							res += "\n\n";

						count++;
					}

					return res + "]";
				}
			}
		}
	}
}

#endif // NDARRAY_TO_STRING