#include <librapid/librapid.hpp>

namespace librapid::detail {
	PreMain::PreMain() {
#if defined(LIBRAPID_WINDOWS)
		// Force the terminal to accept ANSI characters
		system(("chcp " + std::to_string(CP_UTF8)).c_str());
#endif // LIBRAPID_WINDOWS
	}

	// Call the constructor of PreMain to run the pre-main code
	PreMain preMain = PreMain();
} // namespace librapid::detail
