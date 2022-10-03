#pragma once

namespace librapid {
	struct ConsoleSize {
		i16 rows, cols;
	};

#if defined(LIBRAPID_OS_WINDOWS)
	LR_INLINE ConsoleSize getConsoleSize() {
		static CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		i16 cols, rows;
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
