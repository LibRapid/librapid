#ifndef LIBRAPID_TIME_UTILS
#define LIBRAPID_TIME_UTILS

#include <chrono>
#include <librapid/config.hpp>

namespace librapid {
	/**
	 * \rst
	 *
	 * Get the number of seconds since the Unix Epoch in a ``double`` format
	 *
	 * \endrst
	 */
	double seconds();

	/**
	 * \rst
	 *
	 * Pause program execution for a specific number of seconds.
	 *
	 * \endrst
	 */
	void sleep(double s);
}

#endif // LIBRAPID_TIME_UTILS