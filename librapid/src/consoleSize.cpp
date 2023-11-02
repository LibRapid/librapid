#include <librapid/librapid.hpp>

#if defined(LIBRAPID_OSX)
#	include <unistd.h>
#	include <sys/ioctl.h>
#elif defined(LIBRAPID_LINUX)
#	include <unistd.h>
#	include <sys/ioctl.h>
#elif defined(LIBRAPID_WINDOWS)
#	include <windows.h>
#endif

namespace librapid {
	ConsoleSize consoleSize() {
#if defined(LIBRAPID_OSX)
		struct winsize w {};
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
		return {w.ws_row, w.ws_col};
#elif defined(LIBRAPID_LINUX)
		struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
		return {w.ws_row, w.ws_col};
#elif defined(LIBRAPID_WINDOWS)
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
		return {csbi.srWindow.Bottom - csbi.srWindow.Top + 1,
				csbi.srWindow.Right - csbi.srWindow.Left + 1};
#else
		return {24, 80};
#endif
	}
} // namespace librapid
