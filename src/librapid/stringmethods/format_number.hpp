#ifndef LIBRAPID_FORMAT_NUMBER
#define LIBRAPID_FORMAT_NUMBER

#include <librapid/config.hpp>

namespace librapid
{
	std::string format_number(const double &val, bool floating = true);
}

#endif // LIBRAPID_FORMAT_NUMBER