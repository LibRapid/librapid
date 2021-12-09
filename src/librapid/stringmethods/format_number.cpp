#include <librapid/stringmethods/format_number.hpp>
#include <librapid/autocast/custom_complex.hpp>
#include <string>
#include <sstream>
#include <algorithm>

namespace librapid {
	std::string format_number(const double &val, bool floating, bool international) {
		std::stringstream stream;
		stream.precision(10);

		stream << val;

		std::string str = stream.str();

		switch (international) {
			case true:
				if (floating && str.find_last_of('.') == std::string::npos)
					str += ".0";

				return str;
			case false:
				if (floating && str.find_last_of('.') == std::string::npos)
					str += ",0";
				else
					std::replace(str.begin(), str.end(), '.', ',');
				return str;
		}
	}
}