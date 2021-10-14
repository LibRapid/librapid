#ifndef LIBRAPID_CONSOLE_UTILS
#define LIBRAPID_CONSOLE_UTILS

#include <librapid/config.hpp>

// Console width and height information
#if defined(LIBRAPID_OS_WINDOWS)
#include <windows.h>
#elif defined(LIBRAPID_OS_UNIX)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace librapid
{
	struct consoleSize
	{
		lr_int rows;
		lr_int cols;
	};

	consoleSize getConsoleSize();
}

#endif // LIBRAPID_CONSOLE_UTILS