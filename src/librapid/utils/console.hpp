#pragma once

#include "../internal/config.hpp"
#include "../math/vector.hpp"

#if defined(LIBRAPID_OS_WINDOWS)
#	define WIN32_LEAN_AND_MEAN
#	include <Windows.h>
#elif defined(LIBRAPID_OS_UNIX)
#	include <sys/ioctl.h>
#	include <unistd.h>
#endif

namespace librapid {
	struct ConsoleSize {
		int rows, cols;
	};

#if defined(LIBRAPID_OS_WINDOWS)
	LR_INLINE ConsoleSize getConsoleSize() {
		static CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		int cols, rows;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo);
		cols = bufferInfo.srWindow.Right - bufferInfo.srWindow.Left + 1;
		rows = bufferInfo.srWindow.Bottom - bufferInfo.srWindow.Top + 1;
		return {rows, cols};
	}
#elif defined(LIBRAPID_OS_UNIX)
	LR_INLINE ConsoleSize getConsoleSize() {
		static struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
		return {w.ws_row, w.ws_col};
	}
#else
	LR_INLINE ConsoleSize getConsoleSize() {
		// Not a clue what this would run on, or how it would be done
		// correctly, so just return some arbitrary values...
		return {30, 120};
	}
#endif
} // namespace librapid
