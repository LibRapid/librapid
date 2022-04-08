#ifndef LIBRAPID_FORMAT_NUMBER
#define LIBRAPID_FORMAT_NUMBER

// #include <librapid/config.hpp>
#include <sstream>
#include <string>

namespace librapid {
	template<typename T>
	std::string
	format_number(const T &val,
				  bool floating = std::is_floating_point<T>::value) {
		std::stringstream stream;
		stream.precision(10);
		stream << val;
		std::string str = stream.str();
		if (floating && str.find_last_of('.') == std::string::npos) str += ".";
		return str;
	}
} // namespace librapid

#endif // LIBRAPID_FORMAT_NUMBER