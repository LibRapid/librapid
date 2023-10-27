#ifndef LIBRAPID_UTILS_CONSOLE_SIZE_HPP
#define LIBRAPID_UTILS_CONSOLE_SIZE_HPP

namespace librapid {
    struct ConsoleSize {
		int rows;
		int cols;
	};

	/// \brief Get the size of the console window in characters (rows and columns).
	/// \return ConsoleSize
    ConsoleSize consoleSize();
} // namespace librapid

#endif // LIBRAPID_UTILS_CONSOLE_SIZE_HPP