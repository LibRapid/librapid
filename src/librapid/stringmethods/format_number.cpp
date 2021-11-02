#include <librapid/stringmethods/format_number.hpp>
#include <librapid/autocast/custom_complex.hpp>
#include <string>
#include <sstream>

namespace librapid
{
	std::string format_number(const double &val, bool floating)
	{
		std::stringstream stream;
		stream.precision(10);

		stream << val;

		std::string str = stream.str();
		if (floating && str.find_last_of('.') == std::string::npos)
			str += ".";

		return str;
	}
}