#ifndef LIBRAPID_FORMAT_NUMBER
#define LIBRAPID_FORMAT_NUMBER

namespace librapid
{
	template<typename T>
	LR_INLINE std::string format_number(const T &val)
	{
		std::stringstream stream;
		stream.precision(10);

		stream << val;

		std::string str = stream.str();
		if (std::is_floating_point<T>::value && str.find_last_of('.') == std::string::npos)
			str += ".";

		return str;
	}
}

#endif // LIBRAPID_FORMAT_NUMBER