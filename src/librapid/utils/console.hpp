#pragma once

namespace librapid {
	struct ConsoleSize {
		i32 rows, cols;
	};

#if defined(LIBRAPID_OS_WINDOWS)
	LR_INLINE ConsoleSize getConsoleSize() {
		static CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		i32 cols, rows;
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

#if defined(LIBRAPID_OS_WINDOWS)
	// Set the cursor's position
	LR_INLINE void setCursorPosition(i16 row, i16 col) {
		COORD pos	   = {row, col};
		HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO screen;

		GetConsoleScreenBufferInfo(console, &screen);
		SetConsoleCursorPosition(console, pos);
	}

	// Move the cursor by a certain amount in the x and y directions
	LR_INLINE void moveCursor(i32 x, i32 y) {
		CONSOLE_SCREEN_BUFFER_INFO screen;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &screen);
		setCursorPosition(screen.dwCursorPosition.X + x, screen.dwCursorPosition.Y + y);
	}

	LR_INLINE void clearConsole() {
		COORD topLeft  = {0, 0};
		HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
		CONSOLE_SCREEN_BUFFER_INFO screen;
		DWORD written;

		GetConsoleScreenBufferInfo(console, &screen);
		FillConsoleOutputCharacterA(
		  console, ' ', screen.dwSize.X * screen.dwSize.Y, topLeft, &written);
		FillConsoleOutputAttribute(console,
								   FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE,
								   screen.dwSize.X * screen.dwSize.Y,
								   topLeft,
								   &written);
		SetConsoleCursorPosition(console, topLeft);
	}
#elif defined(LIBRAPID_OS_UNIX)
	// Set cursor position on unix
	LR_INLINE void setCursorPosition(i32 row, i32 col) { printf("\033[%d;%dH", row, col); }

	// Move the cursor by a certain amount in the x and y directions
	LR_INLINE void moveCursor(i32 x, i32 y) {
		if (x > 0) {
			printf("\033[%dC", x);
		} else if (x < 0) {
			printf("\033[%dD", -x);
		}

		if (y > 0) {
			printf("\033[%dB", y);
		} else if (y < 0) {
			printf("\033[%dA", -y);
		}
	}

	LR_INLINE void clearConsole() {
		// CSI[2J clears screen, CSI[H moves the cursor to top-left corner
		std::cout << "\x1B[2J\x1B[H";
	}
#else
	// Set cursor position on other platforms
	LR_INLINE void setCursorPosition(i32 row, i32 col) {
		// Not a clue what this would run on, or how it would be done
		// correctly, so just do nothing...
		return;
	}

	LR_INLINE void moveCursor(i32 x, i32 y) {
		// Not a clue what this would run on, or how it would be done
		// correctly, so just do nothing...
		return;
	}

	LR_INLINE void clearConsole() {
		// Not a clue what this would run on, or how it would be done
		// correctly, so just print a bunch of newlines...
		for (i32 i = 0; i < 100; i++) { std::cout << std::endl; }
	}
#endif
} // namespace librapid
