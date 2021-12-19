#ifndef LIBRAPID_FORMAT_NUMBER
#define LIBRAPID_FORMAT_NUMBER

#include <librapid/config.hpp>

namespace librapid {
	template<typename T>
	class Complex;

	std::string format_number(const double &val, bool floating = true, bool international = true);
	std::string format_number(const Complex<double> &val, bool floating = true, bool international = true);
}

#endif // LIBRAPID_FORMAT_NUMBER