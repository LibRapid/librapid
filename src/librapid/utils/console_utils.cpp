#include <librapid/utils/console_utils.hpp>

namespace librapid {
#if defined(LIBRAPID_OS_WINDOWS)

	consoleSize getConsoleSize() {
		static CONSOLE_SCREEN_BUFFER_INFO csbi;
		int cols, rows;

		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
		cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
		rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;

		return {rows, cols};
	}

#elif defined(LIBRAPID_OS_UNIX)
	consoleSize getConsoleSize()
	{
		static struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

		return {w.ws_row, w.ws_col};
	}
#else
	consoleSize getConsoleSize()
	{
		// Not a clue what this would run on, or how it would be done
		// correctly, so just return some arbitrary values...

		return {80, 120};
	}
#endif
}