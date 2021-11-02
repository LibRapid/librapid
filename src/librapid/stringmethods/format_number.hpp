#ifndef LIBRAPID_FORMAT_NUMBER
#define LIBRAPID_FORMAT_NUMBER

#include <librapid/config.hpp>
#include <librapid/autocast/custom_complex.hpp>

namespace librapid
{
	std::string format_number(const double &val, bool floating = true);
}

#endif // LIBRAPID_FORMAT_NUMBER